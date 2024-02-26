import torch
# Load via Huggingface Style
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from PIL import Image
import pandas as pd
import pickle
import pdb

# df = pd.read_csv('/home/tangtangwzc/lxmert_clip/HVCR/vqa_comparison_study/v9_unusual_gemini_description_hallucination_statement_flatten_question_0-500_flatten_100.csv')
# df = pd.read_csv('/home/tangtangwzc/lxmert_clip/HVCR/blip_evaluation/hvqa_data.csv')
df = pd.read_csv('/home/tangtangwzc/lxmert_clip/HVCR/blip_evaluation/hvqa_shared_with_zhecan - eval.csv')

# output_filename = 'hvqa_mplug-owl-llama-7b_pred.csv'
output_filename = 'hvqa_mplug-owl-llama-7b_pred_20230223.csv'


pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
)

model.to('cuda')

image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

pred_result_list = []
for idx, row in df.iterrows():
    print("idx: ", idx, ' / ', len(df))
    # image_name = row['image']
    # if '/cns/tp-d/home/golnazg/multi_benchmark/retrieval_uncategorized_2023_0720/' in  row['cns_path']:
    #     path = '/dvmm-filer3a/users/james/gvalue/midjourney_images/' + row['cns_path'].replace('/cns/tp-d/home/golnazg/multi_benchmark/retrieval_uncategorized_2023_0720/', '')
    # else:
    #     path = '/dvmm-filer3a/users/james/gvalue/midjourney_images/' + row['cns_path'].replace('/cns/tp-d/home/golnazg/multi_benchmark/midjourney_images_0427/', '')

    if row['image type'] != 'real':
        path = '/dvmm-filer3a/users/james/gvalue/midjourney_images/'+row['cns_path'].split('/')[-1]
    else:
        path = '/dvmm-filer3a/users/james/gvalue/hvqa_real_images/'+row['url'].split('/')[-1]

    question = row['question']

    # We use a human/AI template to organize the context as a multi-turn conversation.
    # <image> denotes an image placehold.
    prompts = [
    '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: <image>
    Human: {}.
    AI: '''.format(question)]

    # The image paths should be placed in the image_list and kept in the same order as in the prompts.
    # We support urls, local file paths and base64 string. You can custom the pre-process of images by modifying the mplug_owl.modeling_mplug_owl.ImageProcessor
    image_list = [path]

    # generate kwargs (the same in transformers) can be passed in the do_generate()
    generate_kwargs = {
        'do_sample': True,
        'top_k': 5,
        'max_length': 512
    }
    from PIL import Image
    images = [Image.open(_) for _ in image_list]
    inputs = processor(text=prompts, images=images, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    # print(sentence)
    pred_result_list.append(sentence)
    print('Question: ', question)
    print("Answer: ", sentence)

    # pdb.set_trace()
output_dict = {}
output_dict['hvqa_mplug-owl-llama-7b_pred'] = pred_result_list

pdb.set_trace()

# open a file, where you ant to store the data
file = open(output_filename.replace('.csv', '.p'), 'wb')
# dump information to that file
pickle.dump(pred_result_list, file)
# close the file
file.close()

pdb.set_trace()

for column in df.columns:
   output_dict[column] = df[column].tolist()[:len(pred_result_list)]

# output_dict = output_df.to_dict('list')
# for key, value in output_df.items():
    # output_dict[key] = value[:len(list(range(len(return_pred_result_list))))]
pd.DataFrame(output_dict).to_csv(output_filename, index=False)
pdb.set_trace()