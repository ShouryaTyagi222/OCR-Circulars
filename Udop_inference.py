from transformers import AutoProcessor, UdopForConditionalGeneration
from datasets import load_dataset

# load model and processor
processor = AutoProcessor.from_pretrained("nielsr/udop-large", apply_ocr=True)
model = UdopForConditionalGeneration.from_pretrained("nielsr/udop-large")

# load image, along with its words and boxes
dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]

# inference
prompt = "Question answering. In which year is the report made?"
encoding = processor(images=image, text=prompt, return_tensors="pt")
predicted_ids = model.generate(**encoding)
print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])