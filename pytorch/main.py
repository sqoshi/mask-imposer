import argparse
import copy
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
from torchvision import models

from dataset.celeba_dataset import FaceLandmarksDataset
from model import Network

manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "~/datasets/image/celeba/img_align_celeba"
csv_landmarks = "/home/piotr/Documents/mask-imposer/pytorch/dataset/list_landmarks_align_celeba.txt"

workers = 2
batch_size = 1
image_size = 64
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(f"Device: {device}")
########################################################################################
full_dataset = FaceLandmarksDataset(root_dir=dataroot, csv_file=csv_landmarks)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# network = Network()
# network.cpu()
# optimizer = optim.Adam(network.parameters(), lr=0.01)
# criterion = nn.MSELoss()

########################################################################################
# for i in range(10):
#     running_loss = 0
#     for images, labels in trainloader:
#         print(images.shape)
#         print(labels.shape)
#
#         # Training pass
#         optimizer.zero_grad()
#
#         output = network(images)
#         print(output.shape)
#         print(labels.shape)
#         print(output)
#         print(labels)
#         loss = criterion(output, labels)
#
#         # This is where the model learns by backpropagating
#         loss.backward()
#
#         # And optimizes its weights here
#         optimizer.step()
#
#         running_loss += loss.item()
#     else:
#         print("Epoch {} - Training loss: {}".format(i + 1, running_loss / len(trainloader)))

network = Network()
network.cpu()
print(network)
criterion = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.0001)

loss_min = np.inf
num_epochs = 5

start_time = time.time()
for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}... ## {epoch/num_epochs *100}%")

    loss_train = 0
    loss_valid = 0
    running_loss = 0

    network.train()
    for step in range(1, 150 + 1):
        print(f"Step in train_loader: {step} ## {step / 150 * 100}%")
        images, landmarks = next(iter(train_loader))
        images = images.cpu()

        landmarks = landmarks.view(landmarks.size(0), -1).cpu()

        predictions = network(images)

        # print("clear all the gradients before calculating them")
        optimizer.zero_grad()

        # print("find the loss for the current step")
        # print(predictions, landmarks)
        loss_train_step = criterion(predictions.float(), landmarks.float())

        # print("calculate the gradients")
        loss_train_step.backward()

        # print("update the parameters")
        optimizer.step()

        loss_train += loss_train_step.item()
        running_loss = loss_train / step

    # print("Network evaluate")
    network.eval()
    with torch.no_grad():

        for step in range(1, 150 + 1):
            print(f"Step in test_loader: {step} ## {step/150 *100}%")
            images, landmarks = next(iter(test_loader))

            images = images.cpu()
            landmarks = landmarks.view(landmarks.size(0), -1).cpu()

            predictions = network(images)

            # find the loss for the current step
            loss_valid_step = criterion(predictions, landmarks)

            loss_valid += loss_valid_step.item()
            running_loss = loss_valid / step

    loss_train /= len(train_loader)
    loss_valid /= len(test_loader)

    print('\n--------------------------------------------------')
    print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
    print('--------------------------------------------------')

    if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(network.state_dict(), 'face_landmarks.pth')
        print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
        print('Model Saved\n')

print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time() - start_time))
