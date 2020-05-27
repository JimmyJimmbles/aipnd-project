import argparse
import time
import torch
import helpers
from helpers import load_data, label_mapping, network_setup, train_network, save_nn_checkpoint, load_nn_checkpoint, predict, validation

# init parser
parse = argparse.ArgumentParser(description="Settings for NN")

# Command Line args, using refernece from project 1 and github
parse.add_argument('--model', default="vgg16",
                   type=str, help='NN model to use')
parse.add_argument('--learning_rate', default=0.001,
                   type=float, help='Learning Rate for Training')
parse.add_argument('--file_name', default='network_checkpoint.pth',
                   type=str, help='Directory in witch to save the NN checkpoints.')
parse.add_argument('--hidden_layers', default=[2000, 500, 250], type=list,
                   help='This is a multiple layer network and needs a list of hidden layers')
parse.add_argument('--epochs', default=1, type=int,
                   help='The amount of time you want run through the NN training')
parse.add_argument('--gpu', default='cpu', type=str, help='Use CPU or GPU')

# parse the argument to get variables
parse_args = parse.parse_args()
model = parse_args.model
learning_rate = parse_args.learning_rate
hidden_layers = parse_args.hidden_layers
epochs = parse_args.epochs
device = parse_args.gpu
file_name = parse_args.file_name

# Load all the data and models
dataloaders, image_datasets = load_data('./flowers')
nn_model, criterion, optimizer = network_setup(
    model, hidden_layers, learning_rate, device)

nn_model.class_to_idx = image_datasets['training'].class_to_idx
model_dict = {
    'arch': 'vgg16',
    'epochs': epochs,
    'classifier': nn_model.classifier,
    'optimizer_state': optimizer.state_dict,
    'state_dict': nn_model.state_dict(),
    'class_to_idx': nn_model.class_to_idx
}
file_name = 'network_checkpoint.pth'

# Time to train!
start_time = time.time()
print("Start: {}".format(start_time))
train_network(nn_model, dataloaders, optimizer, epochs, 30, device)

# Save the network check point
save_nn_checkpoint(model_dict, file_name)

with torch.no_grad():
    loss, accuracy = validation(nn_model, dataloaders['testing'], criterion)

print("Accuracy with testing data: {:.2f}%".format(
    (accuracy / len(dataloaders['testing'])) * 100))

print("End. Total Time: {}".format(time.time() - start_time))
