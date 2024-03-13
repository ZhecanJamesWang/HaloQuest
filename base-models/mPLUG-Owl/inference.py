import torch
# Load via Huggingface Style
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from PIL import Image
import pandas as pd
import pickle
import pdb

df = pd.read_csv('input.csv')

output_filename = 'output.csv'

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

    path = row['img_path']
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
    pred_result_list.append(sentence)
    print('Question: ', question)
    print("Answer: ", sentence)

output_dict = {}
output_dict['pred_ans'] = pred_result_list

pdb.set_trace()

# open a file, where you ant to store the data
file = open(output_filename.replace('.csv', '.p'), 'wb')
# dump information to that file
pickle.dump(pred_result_list, file)
# close the file
file.close()

for column in df.columns:
   output_dict[column] = df[column].tolist()[:len(pred_result_list)]

pd.DataFrame(output_dict).to_csv(output_filename, index=False)