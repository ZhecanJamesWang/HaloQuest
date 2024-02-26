import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from PIL import Image
import random
import math
import pdb
import pandas as pd
import pickle

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Added by Haoxuan
import pdb
import jsonlines
from copy import deepcopy
import yaml
import re

VCR_INPUT_PROMPT = '''[placeholder]
Can you find out the more likely correct answer from the above four answer choices? Kindly demonstrate your reasoning and inference process as you formulate your response.
'''
VCR_DECISION_PROMPT = '''
If you have found the more likely answer, directly reply with the answer id in the format of “Predicted Answer: [id of correct answer]” without any other words, where the id ranges from 1 to 4. Otherwise reply with “We are not sure which option is correct”. 
'''
# VCR_DECISION_PROMPT = '''
# If you have found the more likely answer, summarize it and directly reply with “Predicted Answer: [correct answer]”. Otherwise reply with “We are not sure”.
# '''

detail_describe_instructions = [
    "Describe the following image in detail.",
    "Provide a detailed description of the given image.",
    "Give an elaborate explanation of the image you see.",
    "Share a comprehensive rundown of the presented image.",
    "Offer a thorough analysis of the image.",
    "Explain the various aspects of the image before you.",
    "Clarify the contents of the displayed image with great detail.",
    "Characterize the image using a well-detailed description.",
    "Break down the elements of the image in a detailed manner.",
    "Walk through the important details of the image.",
    "Portray the image with a rich, descriptive narrative.",
    "Narrate the contents of the image with precision.",
    "Analyze the image in a comprehensive and detailed manner.",
    "Illustrate the image through a descriptive explanation.",
    "Examine the image closely and share its details.",
    "Write an exhaustive depiction of the given image.",
]

concise_describe_instructions = [
    "Describe the following image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the following image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo below.",
    "Write a terse but informative summary of the following picture.",
    "Create a compact narrative representing the image presented.",
]

prompt_pool = detail_describe_instructions + concise_describe_instructions

prompt_pool = [ "Describe the following image in detail."]


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)

def load_vcr_data(vcr_data_path):
    img_paths = {}
    vcr_anno_path = os.path.join(vcr_data_path, 'val.jsonl')

    # Obtain subset annot_ids
    id_partitions = None
    if args.data_subset is not None:
        with open(args.data_subset, "r") as file:
            id_partitions = yaml.safe_load(file)
            
    # Obtain partition ids, either in number of annot_id
    # id_partitions = None
    # num_partitions = None
    # if '/' in args.data_partition:
    #     with open(args.data_partition, "r") as file:
    #         id_partitions = yaml.safe_load(file)
    # else:
    #     assert '_' in args.data_partition
    #     start_data_id, end_data_id = args.data_partition.split('_')
    #     num_partitions = [i for i in range(int(start_data_id), int(end_data_id)+1)]

    with jsonlines.open(vcr_anno_path) as reader:
        for ind, cur_ann in enumerate(tqdm(reader)):
            annot_id = cur_ann['annot_id']
            # Only keep the input partition.
            if id_partitions is not None:
                if annot_id not in id_partitions:
                    continue

            img_path = os.path.join(vcr_data_path, 'vcr1images', cur_ann['img_fn'])
            img_paths[annot_id] = img_path

    # Only keep samples in partitions.
    if args.data_partition is not None:
        start_data_id, end_data_id = args.data_partition.split('_')
        _ids = list(img_paths)[int(start_data_id):int(end_data_id)+1]
        img_paths = {key:img_paths[key] for key in _ids}
    
    # _ids = list(img_paths)[int(start_data_id):int(end_data_id)+1]
    # filter_img_paths = {key:img_paths[key] for key in _ids}
    # return filter_img_paths
    return img_paths

def load_okvqa_data(question_path, image_path):
    img_paths = {}
    dataset_question = json.load(open(question_path, "rb"))['questions']

    # Obtain subset annot_ids
    id_partitions = None
    if args.data_subset is not None:
        with open(args.data_subset, "r") as file:
            id_partitions = yaml.safe_load(file)

    # Obtain partition ids, either in number of annot_id
    # id_partitions = None
    # num_partitions = None
    # if '/' in args.data_partition:
    #     with open(args.data_partition, "r") as file:
    #         id_partitions = yaml.safe_load(file)
    # else:
    #     assert '_' in args.data_partition
    #     start_data_id, end_data_id = args.data_partition.split('_')
    #     num_partitions = [i for i in range(int(start_data_id), int(end_data_id)+1)]

    for index_i, question_info_i in enumerate(tqdm(dataset_question)):
        image_id = question_info_i['image_id']
        image_name = 'COCO_val2014_' + str(image_id).zfill(12) + '.jpg'
        question_id = question_info_i['question_id'] 

        # Only keep the input partition.
        if id_partitions is not None:
            if question_id not in id_partitions:
                continue

        img_path = os.path.join(image_path, image_name)
        img_paths[question_id] = img_path

    # Only keep samples in partitions.
    if args.data_partition is not None:
        start_data_id, end_data_id = args.data_partition.split('_')
        _ids = list(img_paths)[int(start_data_id):int(end_data_id)+1]
        img_paths = {key:img_paths[key] for key in _ids}

    # start_data_id, end_data_id = args.data_partition.split('_')
    # _ids = list(img_paths)[int(start_data_id):int(end_data_id)+1]
    # filter_img_paths = {key:img_paths[key] for key in _ids}
    # return filter_img_paths
    return img_paths

# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

def decode_output_text(output_ids, input_ids, tokenizer, conv, args):
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] Sample: {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # pdb.set_trace()
    full_history = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    if args.conv_mode == 'simple_legacy' or args.conv_mode == 'simple':
        while True:
            cur_len = len(outputs)
            outputs = outputs.strip()
            for pattern in ['###', 'Assistant:', 'Response:']:
                if outputs.startswith(pattern):
                    outputs = outputs[len(pattern):].strip()
            if len(outputs) == cur_len:
                break

    try:
        index = outputs.index(conv.sep)
    except ValueError:
        outputs += conv.sep
        index = outputs.index(conv.sep)

    outputs_response = outputs[:index].strip()
    # Return both response and full_chat history.
    return outputs_response, full_history          


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.mm_projector is None:
        patch_config(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.model.vision_tower[0]
        vision_tower.to(device='cuda', dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    else:
        # in case of using a pretrained model with only a MLP projector weights
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower, torch_dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]


    # ========================================
    #             Data Loading
    # ========================================
    # if 'vcr' in args.dataset:
    #     vcr_data_path = '/home/haoxuan/data/vcr1/'
    #     img_paths = load_vcr_data(vcr_data_path)
    # elif 'okvqa' in args.dataset:
    #     okvqa_question_path = '/dvmm-filer3a/users/rui/multi-task/datasets/okvqa/OpenEnded_mscoco_val2014_questions.json'
    #     okvqa_image_path = '/dvmm-filer3a/users/rui/multi-task/coco/val2014/val2014/'
    #     img_paths = load_okvqa_data(question_path=okvqa_question_path, image_path=okvqa_image_path)
    # else:
    #     raise NotImplementedError('Not support other datasets yet.')
    # print(f'{len(img_paths)} Samples from {args.dataset} Loading Finished')

    # df = pd.read_csv('/home/tangtangwzc/lxmert_clip/HVCR/vqa_comparison_study/v9_unusual_gemini_description_hallucination_question_flatten_100.csv')
    # output_filename = 'v9_unusual_gemini_description_hallucination_question_flatten_100_llava_pred.csv'

    # df = pd.read_csv('/home/tangtangwzc/lxmert_clip/HVCR/vqa_comparison_study/v9_unusual_gemini_description_hallucination_statement_flatten_question_0-500_flatten_100.csv')
    # output_filename = 'v9_unusual_gemini_description_hallucination_statement_question_0-500_flatten_100_llava_pred.csv'

    # df = pd.read_csv('/home/tangtangwzc/lxmert_clip/HVCR/blip_evaluation/hvqa_data.csv')
    df = pd.read_csv('/home/tangtangwzc/lxmert_clip/HVCR/blip_evaluation/hvqa_shared_with_zhecan - eval.csv')

    # output_filename = 'hvqa_llava_pred.csv'
    output_filename = 'hvqa_llava_pred_20230223.csv'

    print('{} Samples Loading Finished'.format(len(df)))


    if args.data_subset is not None:
        subset_name = args.data_subset.split('/')[-1].split('.yaml')[0]
    else:
        subset_name = 'fullset'
    result_folder = os.path.join('./result/caption', args.dataset, f'{subset_name}_prompt{args.version}_temp{args.temperature}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    pred_result_list = []
    result = {}
    temperature = args.temperature
    for idx, row in df.iterrows():
        print("idx: ", idx, ' / ', len(df))
        # image_name = row['image']
        # img_name = row['cns_path'].replace('/cns/tp-d/home/golnazg/multi_benchmark/midjourney_images_0427/', '')

        # if '/cns/tp-d/home/golnazg/multi_benchmark/retrieval_uncategorized_2023_0720/' in  row['cns_path']:
        #     img_name = row['cns_path'].replace('/cns/tp-d/home/golnazg/multi_benchmark/retrieval_uncategorized_2023_0720/', '')
        # else:
        #     img_name = row['cns_path'].replace('/cns/tp-d/home/golnazg/multi_benchmark/midjourney_images_0427/', '')

        # img_path = '/dvmm-filer3a/users/james/gvalue/midjourney_images/' + img_name

        if row['image type'] != 'real':
            img_path = '/dvmm-filer3a/users/james/gvalue/midjourney_images/'+row['cns_path'].split('/')[-1]
        else:
            img_path = '/dvmm-filer3a/users/james/gvalue/hvqa_real_images/'+row['url'].split('/')[-1]

        question = row['question']
    # for id, img_path in tqdm(img_paths.items(), desc='Generating Captionings'):
    #     image_file = img_path

        # Prepare Prompts of different versions.
        # if args.version == 'v1':
        #     qs = '''Describe this image as detail as possible in one paragraph.'''
        # elif args.version == 'v2':
        #     qs = '''Give a clear and concise summary of the image below.'''
        # elif args.version == 'v3':
        #     qs = '''Give a clear and concise summary of the image below in one paragraph.'''
        
        qs = question

        cur_prompt = qs
        if mm_use_im_start_end:
            qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
        else:
            qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

        if args.conv_mode == 'simple_legacy':
            qs += '\n\n### Response:'
        # conv = default_conversation.copy()
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        image = Image.open(img_path)
        # image = Image.open(os.path.join(args.image_folder, image_file))
        # image.save(os.path.join(save_image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=temperature,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        outputs, full_chat = decode_output_text(output_ids, input_ids, tokenizer, conv=conv, args=args)

        print('question: ', question)
        print('answer: ', outputs)

        result[id] = {
            'img_fn':img_path,
            'caption':outputs,
            'annot_id':id,
            'history':full_chat}
        
        pred_result_list.append(outputs)

        # with open(os.path.join(result_folder, '{}.yaml'.format(id)), 'w') as f:
        #     yaml.dump(result[id], f)


    output_dict = {}
    output_dict['pred_llava'] = pred_result_list

    # pdb.set_trace()

    # open a file, where you ant to store the data
    # file = open('v9_unusual_gemini_description_hallucination_question_flatten_blip2_flanxxl_pred.p', 'wb')
    # file = open('v9_unusual_gemini_description_hallucination_statement_question_0-500_flatten_100_llava_pred.p', 'wb')
    # file = open('hvqa_llava_pred.p', 'wb')
    file = open('hvqa_llava_pred_20230223.p', 'wb')
    
    # dump information to that file
    pickle.dump(pred_result_list, file)
    # close the file
    file.close()

    # pdb.set_trace()

    for column in df.columns:
        output_dict[column] = df[column].tolist()[:len(pred_result_list)]

    # output_dict = output_df.to_dict('list')
    # for key, value in output_df.items():
        # output_dict[key] = value[:len(list(range(len(return_pred_result_list))))]
    pd.DataFrame(output_dict).to_csv(output_filename, index=False)
    pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--data_subset", type=str, default=None, help="specify the subset of the dataset.")
    parser.add_argument("--data_partition", type=str, default=None, help="id_id, specify the partition of the dataset.")
    # parser.add_argument("--data_partition", type=str, default='0_9', help="id_id or path, specify the partition of the dataset.")
    parser.add_argument("--dataset", type=str, default='vcr', help="specify the dataset to generate caption.")
    parser.add_argument("--version", type=str, default='v1', help="specify the version of prompt.")
    args = parser.parse_args()

    eval_model(args)
