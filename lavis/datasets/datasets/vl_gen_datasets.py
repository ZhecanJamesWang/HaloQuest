"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import pandas as pd
import os
import json
import re
from PIL import Image,ImageOps,ImageDraw,ImageFont
import torch
from PIL import ImageFile
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.base_dataset import BaseDataset
import random
from collections import defaultdict

class VL_Gen_dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, ann_paths,image_size,split):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        # super().__init__(vis_processor, text_processor)
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.is_train = 'train' == split
        self.img_size=image_size
        self.num_item = 0 
        self.annotation = []
        use_uqa = True
        if use_uqa:
            self.load_uqa()
    def load_uqa(self):
        df = pd.read_csv("/dvmm-filer3a/users/junzhang/projects/unilm/beit3/datasets/hvqa/hvqa.csv")
        annotation = []
        for row_id, row in df.iterrows():
            if row['image type'] != 'real':
                img_path = '/dvmm-filer3a/users/james/gvalue/midjourney_images/'+row['cns_path'].split('/')[-1]
            else:
                img_path = '/dvmm-filer3a/users/james/gvalue/hvqa_real_images/'+row['url'].split('/')[-1]

            # if '/cns/tp-d/home/golnazg/multi_benchmark/retrieval_uncategorized_2023_0720/' in  row['cns_path']:
            #     img_path = '/dvmm-filer3a/users/james/gvalue/midjourney_images/' + row['cns_path'].replace('/cns/tp-d/home/golnazg/multi_benchmark/retrieval_uncategorized_2023_0720/', '')
            # else:
            #     img_path = '/dvmm-filer3a/users/james/gvalue/midjourney_images/' + row['cns_path'].replace('/cns/tp-d/home/golnazg/multi_benchmark/midjourney_images_0427/', '')
            
            question = row['question']
            try:
                answers = row['groundtruth responses'].split(';')
            except Exception:
                answers = ['']
            if '\u2019' in question:
                question=question.replace('\u2019',"'")
            for idx in range(len(answers)):
                if '\u2019' in answers[idx]:
                    answers[idx]=answers[idx].replace('\u2019',"'")
            datapoint = {'img_path':img_path,
                         'question':question,
                         'answer':answers,
                         'dataset':'uqa',
                         'idx':row_id,
                         'url':row['url']
                         }
            annotation.append(datapoint)
        # shuffler = random.Random(0)
        # shuffler.shuffle(annotation)
        # if self.is_train:
        #     annotation=annotation[:1500]
        #     for d in annotation:
        #         for ans in d['answer']:
        #             self.annotation.append({'img_path':d['img_path'],
        #                                     'question':d['question'],
        #                                     'answer':ans,
        #                                     'dataset':'uqa'})
        #     print('train set, total',len(annotation),'datapoints', len(self.annotation),'sentences')
                    
        # else:
        self.annotation+=annotation
        print('val set, total',len(self.annotation),'datapoints')       
            
    def __getitem__(self, index):
        instance = self.annotation[index]
        question = "Question: " + instance['question'] + " Answer: "
        num_ans = 1
        answer = instance['answer']
        weights=[1]
            
        
        img_path = instance['img_path']
        image = Image.open(img_path).convert("RGB")
        w,h = image.size
        if w>h:
            size = (self.img_size,int(self.img_size*(h/w)))
        else:
            size = (int(self.img_size*(w/h)),self.img_size)
        image=image.resize(size)
        new_img = Image.new('RGB',size=(self.img_size,self.img_size))
        new_img.paste(image,(0,0))
        image = self.vis_processor(new_img)
        inputs = question
        # inputs = self.text_processor(question)
        if isinstance(answer,list):
            outputs = [self.text_processor(a.lstrip().rstrip().replace('\n',' ')) for a in answer]
        else:
            outputs = [self.text_processor(answer.lstrip().rstrip().replace('\n',' '))]
        
        return {
            "image": image,
            "text_input": inputs,
            "answers": outputs,
            "img_path":img_path,
            "dataset":instance['dataset'],
            "n_answers":num_ans,
            "weights":weights,
            "question_id": instance['idx'],
            
        }
    def collater(self, samples):
        image_list, question_list, answer_list,img_paths,weights,num_ans,datasets = [], [], [],[],[],[],[]
        question_id_list=[]
        for sample in samples:
            dataset = sample['dataset']
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])

            answer = sample["answers"] if 'answers' in sample.keys() else sample["answer"]
            answer_list.append(answer)
            if 'img_path' in sample.keys():
                img_paths.append(sample['img_path'])
            else:
                img_paths.append('')
            if 'question_id' in sample.keys():
                question_id_list.append(sample['question_id'])
            else:
                question_id_list.append(0)
            weights+=sample['weights']
            num_ans.append(len(answer))
            datasets.append(dataset)
        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answers": answer_list,
            "img_path":img_paths,
            "weight":torch.tensor(weights),
            "n_answers":torch.tensor(num_ans),
            "dataset":datasets,
            "question_id":question_id_list}