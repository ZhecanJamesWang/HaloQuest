from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor
import pandas as pd
import pdb

# df = pd.read_csv("all_model_results.csv")
df = pd.read_csv("hvqa_shared_with_zhecan - eval.csv")

model_metrics = []

model_names = ["BLIP2-T5",
               "beit3"
               "pred_otter-OTTER-9B-LA-InContext_pred",
               "pred_open-flamingo-3B-vitl-mpt1b_pred",
               "pred_instruct-blip-flant5xxl_pred"]

# ['pred_bllp2_flanxxl',
            # 'pred_otter-OTTER-9B-LA-InContext_pred',
                # 'pred_minigpt4','pred_open-flamingo-3B-vitl-mpt1b_pred',
                # 'pred_instruct-blip-flant5xxl_pred']
for model_name in model_names:
    # model_name = 'pred_instruct-blip-flant5xxl_pred'
    gt_row = 'groundtruth responses'
    scorers = [BleuScorer(),CiderScorer()]
    meteor = Meteor()

    pred_for_meteor = dict()
    refs_for_meteor = dict()


    for idx,r in df.iterrows():
        y_true = r[gt_row]
        y_pred = r[model_name]
        if not isinstance(y_pred,str):
            y_pred='NA'
        pred_for_meteor[idx]=[y_pred]
        refs_for_meteor[idx] = y_true
        for scorer in scorers:
            scorer.cook_append(y_true,y_pred)

    pdb.set_trace()
    avg_meteor,meteor_scores = meteor.compute_score(refs_for_meteor,pred_for_meteor) #compute meteor    
    scores = [s.compute_score() for s in scorers] #compute score
    overall_scores,data_scores = [s[0] for s in scores],[s[1]for s in scores] #avg score, individual data score
    avg_bleu1,avg_bleu2,avg_bleu3,avg_bleu4 =overall_scores[0]
    avg_cider = overall_scores[1]

    bleu1,bleu2,bleu3,bleu4 = data_scores[0]
    #zip the scores
    cider = list(data_scores[1])
    all_scores = zip(bleu1,bleu2,bleu3,bleu4,cider,meteor_scores)
    metrics = dict()
    metrics['model_name'] = model_name
    metrics['bleu1'] = avg_bleu1
    metrics['bleu2'] = avg_bleu2
    metrics['bleu3'] = avg_bleu3
    metrics['bleu4'] = avg_bleu4
    metrics['cider'] = avg_cider
    metrics['meteor'] = avg_meteor
    print(metrics)
    model_metrics.append(metrics)
pd.DataFrame(model_metrics).to_csv('results.csv', index=False)