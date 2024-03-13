import argparse
import os
import random
import pdb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import jsonlines
from tqdm import tqdm
import yaml
import json
from os import listdir
from os.path import isfile, join, isdir

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--dataset", type=str, default='retrieval', help="specify the dataset to generate caption.")
    parser.add_argument("--version", type=str, default='v1', help="specify the version of prompt.")
    parser.add_argument("--data_partition", type=str, default='0_9', help="specify the partition of the dataset.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def load_vcr_data(vcr_data_path):
    img_paths = {}
    vcr_anno_path = os.path.join(vcr_data_path, 'val.jsonl')

    with jsonlines.open(vcr_anno_path) as reader:
        for cur_ann in tqdm(reader):
            annot_id = cur_ann['annot_id']
            img_path = os.path.join(vcr_data_path, 'vcr1images', cur_ann['img_fn'])
            img_paths[annot_id] = img_path

    start_data_id, end_data_id = args.data_partition.split('_')
    _ids = list(img_paths)[int(start_data_id):int(end_data_id)+1]
    filter_img_paths = {key:img_paths[key] for key in _ids}
    return filter_img_paths


def load_image_data(path):
    image_ext = ['.png', '.jpg', '.jpeg']
    filenames = [f for f in listdir(path) if (isfile(join(path, f)) and any(ext in f for ext in
                                                                                             image_ext))]
    filename_paths = [join(path, f) for f in listdir(path) if (isfile(join(path, f)) and any(ext in f for ext in
                                                                                             image_ext))]
    subdir_paths = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
    for subdir in subdir_paths:
        sub_filenames = [f for f in listdir(subdir) if isfile(join(subdir, f))]
        sub_filename_paths = [join(subdir, f) for f in listdir(subdir) if isfile(join(subdir, f))]
        filenames.extend(sub_filenames)
        filename_paths.extend(sub_filename_paths)

    start_data_id, end_data_id = args.data_partition.split('_')
    return filenames[int(start_data_id):int(end_data_id)+1], filename_paths[int(start_data_id):int(end_data_id)+1]


def load_okvqa_data(question_path, image_path):
    img_paths = {}
    dataset_question = json.load(open(question_path, "rb"))['questions']
    for index_i, question_info_i in enumerate(tqdm(dataset_question)):
        image_id = question_info_i['image_id']
        image_name = 'COCO_val2014_' + str(image_id).zfill(12) + '.jpg'
        question_id = question_info_i['question_id'] 
        img_path = os.path.join(image_path, image_name)
        img_paths[question_id] = img_path

    start_data_id, end_data_id = args.data_partition.split('_')
    _ids = list(img_paths)[int(start_data_id):int(end_data_id)+1]
    filter_img_paths = {key:img_paths[key] for key in _ids}
    return filter_img_paths

args = parse_args()
# ========================================
#             Data Loading
# ========================================
if 'vcr' == args.dataset:
    vcr_data_path = '...'
    img_paths = load_vcr_data(vcr_data_path)
elif 'uqa' == args.dataset:
    img_names, img_paths = load_image_data('..')
elif 'okvqa' == args.dataset:
    okvqa_question_path = '...'
    okvqa_image_path = '...'
    img_paths = load_okvqa_data(question_path=okvqa_question_path, image_path=okvqa_image_path)
else:
    raise NotImplementedError('Not support other datasets yet.')
print('{} Samples Loading Finished'.format(len(img_paths)))

result_folder = os.path.join('./result/caption', args.dataset, args.version)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)


# ========================================
#             Model Initialization
# ========================================
print('Initializing Chat')
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


# ========================================
#             Generating Captions
# ========================================
result = {}
# for key, img_path in tqdm(img_paths.items(), desc='Generating Captions'):
for idx, img_path in enumerate(tqdm(img_paths, desc='Generating Captions')):
    key = img_names[idx].replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    if os.path.exists(os.path.join(result_folder, '{}.yaml'.format(key))):
        print('FOUND PRE-EXISTING: ')
        print(os.path.join(result_folder, '{}.yaml'.format(key)))
        continue

    if len(key.split('_')) > 2:
        filename_words = []
        for word in key.split('_'):
            if word.isalpha():
                filename_words.append(word)

        filename_words_str = ' '.join(filename_words)
    else:
        filename_words_str = ''

    img_list = []
    chat_state = CONV_VISION.copy()

    if filename_words_str == '':
        text_input = '''Describe this image as detail as possible. Don’t imagine things not existed in the image.'''
    else:
        text_input = '''Describe this image as detail as possible. Don’t imagine things not existed in the image. Here
        are some hint context about the image.'''

        text_input += filename_words_str

    num_beams = 1
    temperature = 1

    try:
        # print('Start extracting image feature')
        llm_message = chat.upload_img(img_path, chat_state, img_list)

        # print('Asking questions')
        chat.ask(text_input, chat_state)

        # print('Generating captions')
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=num_beams,
                                  temperature=temperature,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
        # print('Generated Caption:', llm_message)
        result[key] = {
            'img_fn':img_path,
            'caption':llm_message,
            'annot_id':key,
            'prompt':text_input}

        with open(os.path.join(result_folder, '{}.yaml'.format(key)), 'w') as f:
            yaml.dump(result[key], f)


    except Exception as e:
        print(e)


with open(os.path.join(result_folder, 'all_{}.yaml'.format(args.data_partition)), 'w') as f:
    yaml.dump(result, f)

