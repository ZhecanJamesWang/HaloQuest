"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os

from lavis.common.dist_utils import main_process
import torch.distributed as dist
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.datasets.data_utils import prepare_sample
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
import torch
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor

from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval

@registry.register_task("vl_gen")
class VL_Gen(BaseTask):
    def __init__(self, max_len,cfg):
        super().__init__()

        self.max_len = max_len
        self.report_metric = True
        run_cfg = cfg.run_cfg
        self.output_dir = run_cfg.output_dir


    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        max_len = run_cfg.max_len_input
        return cls(
            max_len,cfg
        )
    def train_step(self, model, samples):
        answers = []
        for a in samples['answers']:
            if isinstance(a,str):
                answers.append(a)
            else:
                answers.extend(a)
        # assert 'uqa' not in samples['dataset']
        samples['answers'] = answers
        output = model(samples)
        # loss_dict = {'loss':output.loss}
        return output["loss"], output
    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        results = {'vqa':[],'uqa':[]}
        # TODO make it configurable
        print_freq = 10

        tot_loss = 0
        num_result = 0
        self.tot_loss_num_loss = torch.zeros((2,),device=model.device)
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            captions,loss = self.valid_step(model=model, samples=samples)
            num_result+=1
            for gen,ref,question,img_path,dataset,question_id in zip(captions,samples['answers'],\
                                                        samples['text_input'],samples['img_path'],\
                                                            samples['dataset'],samples['question_id']):
                # ref = [r.replace('\n',' ') for r in ref]
                results[dataset].append({'answer':gen,
                                'ref':ref,
                                "question":question,
                                "img_path":img_path,
                                'question_id':int(question_id)
                                })
        if is_dist_avail_and_initialized():
            dist.barrier()
        torch.distributed.reduce(self.tot_loss_num_loss,dst=0,op=torch.distributed.ReduceOp.SUM)
        return results

    def valid_step(self, model, samples):
        captions = model.predict_answers(
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=50,
            min_length=1,
        )
        # result = model(samples)
        # loss = result.loss
        return captions,0
    def before_evaluation(self, model, dataset, **kwargs):
        super().before_evaluation(model,dataset=dataset, task_type=type(self))
        self.eval_results = {'vqa':[],'uqa':[]}

    def save_result(self,name,result,epoch):
        save_fn = self.output_dir+name+'_'+str(epoch)+'.json'
        with open(save_fn,'w+') as f:
            json.dump(result,f,indent=4)
        return save_fn
    def save_temp_result(self,result,fn):
        path = fn+str(get_rank())+'.json'
        with open(path,'w+') as f:
            json.dump(result,f)
        dist.barrier()
        result = {'vqa':[],'uqa':[]}
        if is_main_process():
            for rank in range(get_world_size()):
                result_file = fn+str(rank)+'.json'
                res = json.load(open(result_file, "r"))
                for dataset in result.keys():
                    result[dataset]+=res[dataset]
        return result

        
    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        path = self.output_dir+'temp/'
        if is_main_process() and not os.path.exists(path):
            os.makedirs(path)
            
        if is_dist_avail_and_initialized():
            dist.barrier()
        val_result = self.save_temp_result(val_result,path)
        if is_main_process():
            metrics = {'agg_metrics':0}
            fns = {}
            for dataset in val_result.keys():
                save_fn = self.save_result(dataset,val_result[dataset],epoch=epoch)
                fns[dataset]=save_fn
            if len(val_result['uqa'])>0:
                eval_results,metrics = self.uqa_results(
                    val_result=val_result['uqa']
                )
                # self.save_result(
                #     result=eval_results,
                #     epoch=epoch
                # )
            if len(val_result['vqa'])>0:
                vqa_metrics = self.vqa_metrics(fns['vqa'])
                for m in vqa_metrics:
                    metrics[m] = vqa_metrics[m]
        else:
            metrics={"agg_metrics":0}
        return metrics
    
    @main_process
    def vqa_metrics(self, vqa_result_fn, split='val'):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}
        vqa = VQA('/dvmm-filer3a/users/junzhang/projects/LAVIS/storage/coco/annotations/v2_mscoco_val2014_annotations.json',
                  '/dvmm-filer3a/users/junzhang/projects/LAVIS/storage/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json')

        # create vqaEval object by taking vqa and vqaRes
        # n is precision of accuracy (number of places after decimal), default is 2
        
        vqa_result = vqa.loadRes(
            resFile=vqa_result_fn, quesFile='/dvmm-filer3a/users/junzhang/projects/LAVIS/storage/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json'
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
    @main_process
    def uqa_results(self, val_result):
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
        metrics['agg_metrics'] = (self.tot_loss_num_loss[0]/self.tot_loss_num_loss[1]).item()
        eval_results['metrics'] = metrics
        eval_results['outputs'] = val_result
        return eval_results,metrics

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets