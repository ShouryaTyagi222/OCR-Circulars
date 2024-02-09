import transformers
from transformers import AutoTokenizer, AutoProcessor
from transformers import AdamW
from transformers import UdopForConditionalGeneration

from PIL import Image
import numpy as np
import pandas as pd
import os
import json
import argparse
import cv2

from datasets import Dataset
from datasets import Features, Sequence, Value, Array2D, Array3D
import torch
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_udop_data(file,img_dir):
  with open(file, 'r') as json_file:
      data_dict = json.load(json_file)

  custom_dataset=[]
  max_input_ids=max(len(data_dict[i]['input_ids'][0]) for i in range(len(data_dict)))
  max_label=max(len(data_dict[i]['label'][0]) for i in range(len(data_dict)))
  max_bbox=max(len(data_dict[i]['bbox'][0]) for i in range(len(data_dict)))


  # Convert lists back to numpy arrays
  for data in tqdm(data_dict):

    input_ids = np.array(data['input_ids'])
    input_ids=np.hstack([input_ids[0],np.zeros((max_input_ids - len(input_ids[0]),))])
    label = np.array(data['label'])
    label = np.hstack([label[0],np.zeros((max_label - len(label[0]),))])
    bbox = np.array(data['bbox'])
    bbox = np.pad(bbox[0], ((0, max_bbox - len(bbox[0])), (0, 0)), mode='constant', constant_values=0)
    image = cv2.imread(os.path.join(img_dir,data['image']))
    image = cv2.resize(image,(224,224))
    # print(image.shape)

    # Convert numpy arrays to tensors
    input_ids = torch.tensor(input_ids).long()
    label = torch.tensor(label).long()
    bbox = torch.tensor(bbox).long()
    image = torch.tensor(image).float()
    image = image.permute(2, 0, 1)

    data_dict = {
        'input_ids': input_ids,
        'label': torch.tensor(label).long(),  # Convert label to LongTensor
        'bbox': torch.tensor(bbox).long(),    # Convert bbox to LongTensor
        'image': image
    }

    # print(data_dict)

    custom_dataset.append(data_dict)

  features = {
      'input_ids': Sequence(feature=Value(dtype='int64')),
      'bbox': Array2D(dtype="int64", shape=(512, 4)),
      'image': Array3D(dtype="int64", shape=(3, 224, 224)),
      'label': Sequence(feature=Value(dtype='int64')),
  }

  return custom_dataset

def custom_collate(batch):
    elem_type = type(batch[0])
    if elem_type in (int, float):
        return torch.tensor(batch)
    elif elem_type is torch.Tensor:
        return torch.stack(batch, dim=0)
    elif elem_type is list:
        return [custom_collate(samples) for samples in zip(*batch)]
    elif elem_type is dict:
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return default_collate(batch)
    
def main(args):
    image_dir=args.image_dir
    output_dir=os.path.join(args.output_path,args.model)
    model_checkpoint = args.model_checkpoint  # "nielsr/udop-large"
    batch_size = int(args.batch_size)
    input_file=args.input_file
    n_epochs= int(args.epochs)


    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    encoded_dataset=load_udop_data(input_file,image_dir)

    train_dataset, test_dataset = train_test_split(encoded_dataset, test_size=0.2, random_state=42)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate)

    # Print the size of both datasets
    print("Length of Train Set", len(train_dataset))
    print("Length of Test Set", len(test_dataset))

    model = UdopForConditionalGeneration.from_pretrained(model_checkpoint)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    checkpoint_path = os.path.join(output_dir,"checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch

    
    # Log Losses to a file
    with open(os.path.join(output_dir,"losses_udop.txt"), "w") as f:
        f.write("")
    for epoch in range(n_epochs):  
        model.train()
        Loss = 0
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device).to(torch.long)
            bbox = batch["bbox"].to(device)
            image = batch["image"].to(device).to(torch.float)
            label = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, bbox=bbox, labels=label, pixel_values=image)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            Loss = Loss + loss.item()
        Loss = Loss / len(train_dataloader)
        print("Training Loss is Epoch:", epoch, "Loss:", Loss)
        with open(os.path.join(output_dir,"losses_udop.txt"), "a") as f:
            f.write(f"Epoch: {epoch} Train_Loss: {Loss}\n")

        model.eval()
        Test_Loss = 0
        for idx, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].to(device).to(torch.long)
            bbox = batch["bbox"].to(device)
            image = batch["image"].to(device).to(torch.float)
            label = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, bbox=bbox, labels=label, pixel_values=image)
            loss = outputs.loss
            print("Loss:", loss.item())
            Test_Loss = Test_Loss + loss.item()
        
        Test_Loss = Test_Loss / len(test_dataloader)
        # Print the loss
        print("Testing Loss is Epoch:", epoch, "Loss:", Test_Loss)

        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': Loss,
        }
        torch.save(checkpoint, checkpoint_path)

        # Log the loss
        with open(os.path.join("losses_udop.txt"), "a") as f:
            f.write(f"Epoch: {epoch} Test_Loss: {Loss}\n")

    # Save the model
    model.save_pretrained(os.path.join(output_dir,"udop-finetuned"))



def parse_args():
    parser = argparse.ArgumentParser(description="Fine Tune", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_file", type=str, default=None, help="input docvqa json Data")
    parser.add_argument("-d", "--image_dir", type=str, default=None, help="path to the input folder of the image files")
    parser.add_argument("-o", "--output_dir", type=str, default='/model_output/', help="path to the output folder")
    parser.add_argument("-e", "--epochs", type=str, default=100, help="number of epochs for model training")
    parser.add_argument("-b", "--batch_size", type=str, default=6, help="batch_size for data")
    parser.add_argument("-m", "--model_checkpoint", type=str, default=None, help="model checkpoint of pretrained model")
    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)