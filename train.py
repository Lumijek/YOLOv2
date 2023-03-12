import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from  matplotlib import patches
import numpy as np
from pprint import pprint
from tqdm import tqdm
from time import perf_counter as timer

from dataset import *
from loss import Yolov2Loss
from model import YOLOv2

import warnings

warnings.filterwarnings("ignore")
torch.set_printoptions(sci_mode=False)


device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")


model = YOLOv2().to(device)
print(sum(p.numel() for p in model.parameters()))
criterion = Yolov2Loss()
epochs = 160
batch_size = 8
classes = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane', 'bicycle', 'boat', 'chair', 'diningtable', 'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']
dataloader = get_dataset(batch_size)

lr = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)

def train_network(model, optimizer, criterion, epochs, dataloader, device):
    model = nn.DataParallel(model)
    model = model.to(device)
    cycle = 0
    for epoch in tqdm(range(epochs), desc="Epoch"):
        if epoch == 10:
            optimizer.param_groups[0]["lr"] = 8e-3
        if epoch == 60:
            optimizer.param_groups[0]["lr"] = 1e-4
        if epoch == 90:
            optimizer.param_groups[0]["lr"] = 1e-5
        for image, label in tqdm(dataloader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            outputs = model(image)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()
            if cycle % 10 == 0:
                print("Loss:", loss.item())
            if cycle % 200 == 0:
                torch.save(model.state_dict(), "model.pt")
            cycle += 1


if __name__ == '__main__':
    train_network(model, optimizer, criterion, epochs, dataloader, device)


