
from transformers import AutoTokenizer, AutoProcessor

from PIL import Image
import os
import json
from tqdm import tqdm


input_file = '/content/drive/MyDrive/tactful/docvqa_dataset.json'
model_checkpoint = 'nielsr/udop-large'
image_dir = '/content/drive/MyDrive/tactful/SplitCircularsv2/first_page'
output_dir = '/content/drive/MyDrive/udop_output'


custom_dataset = []

banned_files = []

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
processor = AutoProcessor.from_pretrained(model_checkpoint, apply_ocr=True)

original_dataset = json.load(open(input_file))

count = 0

for img_name in tqdm(original_dataset['data']):

    if img_name['image'] in banned_files:
        continue

    # try:

    img_path=os.path.join(image_dir,img_name['image'])
    img=Image.open(img_path)

    prompt = f"Question answering. {img_name['question']}"
    encoding = processor(images=img, text=prompt, return_tensors="pt")

    image=img_name['image']
    input_ids=encoding['input_ids'] #.numpy().tolist()
    boxes=encoding['bbox']
    boxes=boxes.float() #.numpy().tolist()
    labels=img_name['answers'][0][0]
    labels=tokenizer.encode(labels, return_tensors="pt") #.numpy().tolist()

    data_dict = {
        'input_ids': input_ids,
        'label': labels,
        'bbox': boxes,
        'image': image
    }

    # print(data_dict)

    custom_dataset.append(data_dict)
    count += 1

new_data=[]
for d in custom_dataset:
  image=d['image']
  label=d['label'].numpy().tolist()
  bbox=d['bbox'].numpy().tolist()
  input_ids=d['input_ids'].numpy().tolist()
  new_data.append({
     'image':image,
     'label':label,
     'bbox':bbox,
     'input_ids':input_ids})

with open(os.path.join(output_dir,'docvqa_udop.json'), 'w') as json_file:
  json.dump(new_data, json_file)