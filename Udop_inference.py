# sudo apt-get install tesseract-ocr libtesseract-dev
# pip install pytesseract 'git+https://github.com/facebookresearch/detectron2.git' 'git+https://github.com/NielsRogge/transformers.git@add_udop' sentencepiece torch spacy pillow

import torch
from PIL import Image
import spacy
import transformer.models.udop as udop
import argparse

def main(args):
    tokenizer = udop.UdopTokenizer.from_pretrained("ArthurZ/udop")
    config = udop.UdopConfig.from_pretrained("nielsr/udop-large")
    model = udop.UdopModel(config)
    image_processor = udop.UdopImageProcessor()
    processor = udop.UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)

    image = Image.open(args.img_path)
    image = image.resize((224, 224))
    question = args.question
    encoding = processor(image, question, return_tensors="pt", padding=True, truncation=True)

    decoder_inputs_embeds = torch.rand((1, encoding['input_ids'].shape[1], model.config.hidden_size))
    outputs = model(**encoding, decoder_inputs_embeds=decoder_inputs_embeds)
    output_tensor = outputs[0][0]  # Assuming the first tensor is the relevant one
    decoded_output = tokenizer.decode(output_tensor.argmax(dim=-1), skip_special_tokens=True)

    print("Decoded Output:", decoded_output)

    with open('Udop_output.txt','w') as f:
        f.write(str(outputs))

def parse_args():
    parser = argparse.ArgumentParser(description="UDOP INFER", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--img_path", type=str, default=None, help="path to the image")
    parser.add_argument("-q", "--question", type=str, default="OUTPUT", help="question")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)