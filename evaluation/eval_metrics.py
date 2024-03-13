import logging
import json
import os
import sys         
 
# in the sys.path list
sys.path.append('../base-models/BLIP2/') 
from lavis.common.registry import registry
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor

from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
def vqa_metrics(vqa_result_fn):
    metrics = {}
    vqa = VQA('storage/coco/annotations/v2_mscoco_val2014_annotations.json',
                'storage/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json')

    # create vqaEval object by taking vqa and vqaRes
    # n is precision of accuracy (number of places after decimal), default is 2
    
    vqa_result = vqa.loadRes(
        resFile=vqa_result_fn, quesFile='storage/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json'
    )
    vqa_scorer = VQAEval(vqa, vqa_result, n=2)
    logging.info("Start VQA evaluation.")
    vqa_scorer.evaluate()

    # print accuracies
    overall_acc = vqa_scorer.accuracy["overall"]
    metrics["agg_metrics"] = overall_acc

    logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
    logging.info("Per Answer Type Accuracy is the following:")

    for ans_type in vqa_scorer.accuracy["perAnswerType"]:
        logging.info(
            "%s : %.02f"
            % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
        )
        metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

    with open(
        os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
    ) as f:
        f.write(json.dumps(metrics) + "\n")

    return metrics

def generative_metrics(val_result):
    scorers = [BleuScorer(),CiderScorer()]
    meteor = Meteor()
    refs = dict()
    outputs = dict()
    refs_to_save,outputs_to_save = dict(),dict()
    for idx,result in enumerate(val_result):
        pred = result['answer']
        ref = result['ref']
        refs_to_save[idx] = ref
        outputs_to_save[idx] = [pred]
        if ref == ['']:
            continue
        for scorer in scorers:
            scorer.cook_append(pred,ref)
        refs[idx]=ref
        outputs[idx]=[pred]

    with open('uvqa_eval/temp/ref.json','w+') as f:
        json.dump(refs_to_save,f)
    with open('uvqa_eval/temp/outputs.json','w+') as f:
        json.dump(outputs_to_save,f)
        
    avg_meteor,meteor_scores = meteor.compute_score(refs,outputs) #compute meteor    
    scores = [s.compute_score() for s in scorers] #compute score
    overall_scores,data_scores = [s[0] for s in scores],[s[1]for s in scores] #avg score, individual data score
    avg_bleu1,avg_bleu2,avg_bleu3,avg_bleu4 =overall_scores[0]
    avg_cider = overall_scores[1]
    
    bleu1,bleu2,bleu3,bleu4 = data_scores[0]
    #zip the scores
    cider = list(data_scores[1])
    all_scores = zip(bleu1,bleu2,bleu3,bleu4,cider,meteor_scores)
    input_result_tuple = zip(val_result,all_scores)
    eval_results = dict()
    metrics = dict()
    for result,scores in input_result_tuple:
        datapoint_score = dict()
        for name,s in zip(['bleu1','bleu2','bleu3','bleu4','cider','meteor'],scores):
            datapoint_score[name]=s
        result['scores'] = datapoint_score
    metrics['bleu1'] = avg_bleu1
    metrics['bleu2'] = avg_bleu2
    metrics['bleu3'] = avg_bleu3
    metrics['bleu4'] = avg_bleu4
    metrics['cider'] = avg_cider
    metrics['meteor'] = avg_meteor
    metrics['agg_metrics'] = avg_cider
    eval_results['metrics'] = metrics
    eval_results['outputs'] = val_result
    return eval_results,metrics