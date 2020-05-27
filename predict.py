import argparse
import torch
import time
import helpers
from helpers import load_data, label_mapping, load_nn_checkpoint, predict

# init parser
parse = argparse.ArgumentParser(description="Image Classifier Predictions")

# Command Line args, using refernece from project 1 and github
parse.add_argument('--image_path', default='./flowers/test/16/image_06657.jpg',
                   type=str, help='The image to input to the NN')
parse.add_argument('--category_names', default='cat_to_name.json',
                   type=str, help='Mapping of categories to real names')
parse.add_argument('--checkpoint', default='network_checkpoint.pth',
                   type=str, help='Checkpoint to start network at')
parse.add_argument('--top_k', default=5, type=int)

# parse the argument to get variables
parse_args = parse.parse_args()
image_path = parse_args.image_path
category_names = parse_args.category_names
checkpoint = parse_args.checkpoint
top_k = parse_args.top_k

# Load the NN from the checkpoin
nn_model = load_nn_checkpoint(checkpoint)

# get categories
cat_to_name = label_mapping(category_names)

ps, labels = predict(image_path, nn_model, top_k)

label_names = [cat_to_name[x] for x in labels]

for x in range(len(label_names)):
    print("Flower: {} Probability: {:0.2f}%".format(
        label_names[x], ps[x] * 100))
