import transformers
from transformers import LayoutLMv2ImageProcessor, LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2ForTokenClassification
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import AdamW

from PIL import Image
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import argparse

from datasets import Dataset
from datasets import Features, Sequence, Value, Array2D, Array3D
import torch
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split


def convert_to_custom_format(original_dataset, banned_files, image_dir):
    custom_dataset = []

    count = 0

    for doc_id, document in enumerate(original_dataset):
        # File Name
        file_name = document["file_name"]

        # Load The File
        image = cv2.imread(os.path.join(image_dir,file_name))

        # Skip if the file is in the banned files
        if file_name in banned_files:
            continue

        try:
            for qa_id, qa_pair in enumerate(document["q_and_a"]):
                boxes_arr = np.array(document["boxes"])
                # Pad the boxes array to 512
                padded_boxes = np.pad(boxes_arr, ((0, 512 - len(boxes_arr)), (0, 0)), mode='constant', constant_values=0)
                # # Get the Channels first
                bbox = boxes_arr  # Placeholder for bbox
                image_tensor = torch.tensor(image).clone().detach()
                image_tensor = image_tensor.permute(2, 0, 1)

                # Fill in your data processing logic here to populate input_ids, bbox, attention_mask, token_type_ids, and image
                input_ids = np.array(qa_pair.get("input_ids", -1))
                # Just take the first 512 tokens
                input_ids = input_ids[:512]

                # Fill in your data processing logic here to populate input_ids, bbox, attention_mask, token_type_ids, and image

                start_positions = qa_pair.get("start_idx", -1)
                end_positions = qa_pair.get("end_idx", -1)

                if start_positions > 512:
                    start_positions = -1
                    continue

                if end_positions > 512:
                    end_positions = -1
                    continue

                custom_example = {
                    'input_ids': input_ids,
                    'bbox': padded_boxes,
                    'image': image_tensor,
                    'start_positions': start_positions,
                    'end_positions': end_positions,
                }

                custom_dataset.append(custom_example)
                count += 1
        except Exception as e:
            print(f"Error processing Document {doc_id}, QA {qa_id}: {str(e)}")
            count += 1
            continue

    features = {
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
        'start_positions': Value(dtype='int64'),
        'end_positions': Value(dtype='int64'),
    }

    return custom_dataset

def custom_collate(batch):
    elem_type = type(batch[0])

    if elem_type in (int, float):
        return torch.tensor(batch)
    elif elem_type is torch.Tensor:
        return torch.stack(batch, dim=0)
    elif elem_type is list:
        # Handle lists differently, especially sequences
        return [custom_collate(samples) for samples in zip(*batch)]
    elif elem_type is dict:
        # Handle dictionaries
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        # For other types, use the default_collate behavior
        return default_collate(batch)
    
def main(args):
    image_dir=args.image_dir
    output_dir=os.path.join(args.output_path,args.model)
    model_checkpoint = args.model_checkpoint
    batch_size = int(args.batch_size)
    banned_txt_path = args.banned_txt_path
    input_file=args.input_file
    n_epochs= int(args.epochs)


    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # A Text file contains all the files that are not to be processed
    with open(banned_txt_path) as f:
        banned_files = f.readlines()

    banned_files = [x.strip() for x in banned_files]

    encoded_dataset = json.load(open(input_file))
    encoded_dataset = convert_to_custom_format(encoded_dataset,banned_files,image_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.decode(encoded_dataset[0]["input_ids"])

    train_dataset, test_dataset = train_test_split(encoded_dataset, test_size=0.2, random_state=42)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate)

    # Print the size of both datasets
    print("Length of Train Set", len(train_dataset))
    print("Length of Test Set", len(test_dataset))

    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    checkpoint_path = os.path.join(output_dir,"checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch

    if args.model=='LayoutLMv2':
        # Log Losses to a file
        with open(os.path.join(output_dir,"losses_lmv2_combined.txt"), "w") as f:
            f.write("")
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            model.train()
            Loss = 0
            for idx, batch in enumerate(train_dataloader):
                # get the inputs;
                input_ids = batch["input_ids"].to(device)
                bbox = batch["bbox"].to(device)
                image = batch["image"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(input_ids=input_ids, bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss
                print("Loss:", loss.item())
                loss.backward()
                optimizer.step()
                Loss = Loss + loss.item()
            Loss = Loss / len(train_dataloader)
            # Print the loss
            print("Training Loss is Epoch:", epoch, "Loss:", Loss)

            with open(os.path.join(output_dir,"losses_lmv2_combined.txt"), "a") as f:
                f.write(f"Epoch: {epoch} Train_Loss: {Loss}\n")

            model.eval()
            Test_Loss = 0
            for idx, batch in enumerate(test_dataloader):
                # get the inputs;
                input_ids = batch["input_ids"].to(device)
                bbox = batch["bbox"].to(device)
                image = batch["image"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                # forward + backward + optimize
                outputs = model(input_ids=input_ids, bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
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
            with open(os.path.join("losses_lmv2_combined.txt"), "a") as f:
                f.write(f"Epoch: {epoch} Test_Loss: {Loss}\n")

        # Save the model
        model.save_pretrained(os.path.join(output_dir,"layoutlmv2b-finetuned"))
    
    elif args.model=='LayoutLMv3':
        # Log Losses to a file
        with open(os.path.join(output_dir,"losses_lmv3_combined.txt"), "w") as f:
            f.write("")
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            model.train()
            Loss = 0
            for idx, batch in enumerate(train_dataloader):
                # get the inputs;
                input_ids = batch["input_ids"].to(device)
                bbox = batch["bbox"].to(device)
                image = batch["image"].to(device, dtype=torch.float)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=image, start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss
                print("Loss:", loss.item())
                loss.backward()
                optimizer.step()
                Loss = Loss + loss.item()
            Loss = Loss / len(train_dataloader)
            # Print the loss
            print("Training Loss is Epoch:", epoch, "Loss:", Loss)

            with open(os.path.join(output_dir,"losses_lmv3_combined.txt"), "a") as f:
                f.write(f"Epoch: {epoch} Train_Loss: {Loss}\n")

            model.eval()
            Test_Loss = 0
            for idx, batch in enumerate(test_dataloader):
                # get the inputs;
                input_ids = batch["input_ids"].to(device)
                bbox = batch["bbox"].to(device)
                image = batch["image"].to(device, dtype=torch.float)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                # forward + backward + optimize
                outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=image, start_positions=start_positions, end_positions=end_positions)
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
            with open(os.path.join("losses_lmv3_combined.txt"), "a") as f:
                f.write(f"Epoch: {epoch} Test_Loss: {Loss}\n")

        # Save the model
        model.save_pretrained(os.path.join(output_dir,"layoutlmv3-finetuned"))
    else:
        print('ENTER AN APPROPRIATE MODEL NAME')



def parse_args():
    parser = argparse.ArgumentParser(description="Fine Tune", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--model", type=str, default=None, help="LayoutLMv2/LayoutLMv3")
    parser.add_argument("-i", "--input_file", type=str, default=None, help="input docvqa json Data")
    parser.add_argument("-d", "--image_dir", type=str, default=None, help="path to the input folder of the image files")
    parser.add_argument("-o", "--output_dir", type=str, default='/model_output/', help="path to the output folder")
    parser.add_argument("-h", "--banned_txt_path", type=str, default=None, help="hindi image files txt path")
    parser.add_argument("-e", "--epochs", type=str, default=100, help="number of epochs for model training")
    parser.add_argument("-b", "--batch_size", type=str, default=6, help="batch_size for data")
    parser.add_argument("-m", "--model_checkpoint", type=str, default=None, help="model checkpoint of pretrained model")
    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)