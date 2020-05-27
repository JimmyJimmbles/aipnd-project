# ------------------------------------------------------------------------------- #
# Helper functions
# define all functions that will be useful for this project
#
# Inspriered by: https://github.com/fotisk07/
# ------------------------------------------------------------------------------- #

# Imports
import PIL
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
import json

# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()


def load_data(data_dir='./flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Set transformation vars
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_size = 224
    max_batch_size = 32

    # Set data transformation dict to house all transforms
    data_transforms = {
        'training': transforms.Compose([transforms.RandomRotation(60),
                                        transforms.RandomResizedCrop(img_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, std)]),
        'validation': transforms.Compose([transforms.Resize(img_size + max_batch_size),
                                          transforms.CenterCrop(img_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, std)]),
        'testing': transforms.Compose([transforms.Resize(img_size + max_batch_size),
                                       transforms.CenterCrop(img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(means, std)])
    }

    # Set image datasets dict to house all transforms
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
    }

    # Set all data loaders
    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=max_batch_size, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=max_batch_size, shuffle=True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=max_batch_size, shuffle=True)
    }

    return dataloaders, image_datasets


def label_mapping(file_name):
    with open(file_name, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def network_setup(model='vgg16', hidden_layers=[2000, 500, 250], lr=0.001):
    if type(model) == type(None):
        print("No model was provided, defaulting to vgg16.")
        nn_model = models.vgg16(pretrained=True)

    if model == 'vgg16':
        nn_model = models.vgg16(pretrained=True)
    elif model == 'densenet121':
        nn_model = models.densenet121(pretrained=True)
    else:
        exec("nn_model = models.{}(pretrained = True)".format(model))

    # Freeze parameters
    for param in nn_model.parameters():
        param.requires_grad = False

    cat_to_name = label_mapping('cat_to_name.json')
    output_len = len(cat_to_name)
    input_len = nn_model.classifier[0].in_features

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_len, hidden_layers[0])),
                                            ('relu1', nn.ReLU()),
                                            ('drop1', nn.Dropout(0.2)),
                                            ('fc2', nn.Linear(
                                                hidden_layers[0], hidden_layers[1])),
                                            ('relu2', nn.ReLU()),
                                            ('drop2', nn.Dropout(0.15)),
                                            ('fc3', nn.Linear(
                                                hidden_layers[1], hidden_layers[2])),
                                            ('relu3', nn.ReLU()),
                                            ('drop3', nn.Dropout(0.1)),
                                            ('fc4', nn.Linear(
                                                hidden_layers[2], output_len)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
    # Set the new classifier to the network
    nn_model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(nn_model.classifier.parameters(), lr)

    nn_model.to(device)

    return nn_model, criterion, optimizer


def validation(model, dataloader, criterion):
    loss = 0
    accuracy = 0

    for ii, (inputs, labels) in enumerate(dataloader):
        # Set inputs and labels based on device
        inputs, labels = inputs.to(device), labels.to(device)

        # forward pass to get our log probs
        log_ps = model.forward(inputs)
        batch_loss = criterion(log_ps, labels)

        loss += batch_loss.item()

        # calc the accuracy
        ps = torch.exp(log_ps)
        # get top 1 from topk
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return loss, accuracy


def train_network(nn_model, dataloaders, optimizer, epochs=5, print_every=30):
    steps = 0

    # get dataloaders
    if type(dataloaders) == type(None):
        dataloaders, image_datasets = load_data('./flowers')

    print("Traning has started, this may take a while...")

    for e in range(epochs):
        running_loss = 0

        for inputs, labels in dataloaders['training']:
            steps += 1

            # Set inputs and labels based on device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Get log probabbilities from forward pass
            logps = nn_model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                nn_model.eval()

                with torch.no_grad():
                    validation_loss, accuracy = validation(
                        nn_model, dataloaders['validation'], criterion)

                print("Epoch: {}/{}".format(e + 1, epochs),
                      "Training Loss: {:.3f}".format(
                          running_loss / print_every),
                      "Validation Loss: {:.3f}".format(
                          validation_loss / len(dataloaders['validation'])),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(dataloaders['validation'])))

                running_loss = 0
                nn_model.train()

    print("Traning Ended!")


def save_nn_checkpoint(model_dict, file_name):
    """ Helper funciton to quickly save the nn """

    torch.save(model_dict, file_name)


def load_nn_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    hidden_layers = [2000, 500, 250]
    lr = 0.001

    nn_model, criterion, optimizer = network_setup('vgg16', hidden_layers, lr)

    nn_model.load_state_dict(checkpoint['state_dict'])

    nn_model.classifier = checkpoint['classifier']

    nn_model.class_to_idx = checkpoint['class_to_idx']

    return nn_model


def process_image(image):
    # Set transformation vars
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_size = 224
    max_batch_size = 32

    pil_image = PIL.Image.open(image)

    image_transforms = transforms.Compose([transforms.Resize(img_size + max_batch_size),
                                           transforms.CenterCrop(img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, std)])

    image_tensor = image_transforms(pil_image)

    return image_tensor


def predict(image_path, model, topk=5):
    # answer found here: https://stackoverflow.com/questions/9777783/suppress-scientific-notation-in-numpy-when-creating-array-from-nested-list
    np.set_printoptions(suppress=True, formatter={
                        'float_kind': '{:0.4f}'.format})

    # set model to inference mode and use cpu
    model.to('cpu')
    model.eval()

    image_tensor = process_image(image_path)
    # fix for dim found here: https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    image_tensor = image_tensor.unsqueeze_(0)

    with torch.no_grad():
        log_ps = model.forward(image_tensor)

    # get top probabilties and labels
    ps = torch.exp(log_ps)
    ps_top, labels_top = ps.topk(topk)

    # array for probabilties and labels used for mapping labels to their string name
    ps_top_arr = np.array(ps_top)[0]
    labels_top_arr = np.array(labels_top[0])

    class_to_idx = model.class_to_idx
    idx_to_class = {x: y for y, x in class_to_idx.items()}

    label_list = []

    for x in labels_top_arr:
        label_list += [idx_to_class[x]]

    return ps_top_arr, label_list
