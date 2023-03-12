import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
from pprint import pprint
from time import perf_counter as pf

classes = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane', 'bicycle', 'boat', 'chair', 'diningtable', 'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']

class YoloDataset(Dataset):
    def __init__(self, data, S=13, B=5, C=20, image_size=416):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.data = data
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # output (B, S, S, B * 5 + C)
        # where last dimension is like
        # [x, y, w, h, I(O), 0, 0, 0, 0, 0 Pr(C0), Pr(C1) ... Pr(19)] # 0s so that output shape and label shape matchup to 30
        # I(O) is binary indicator function if object is present

        i, o = self.data[idx]

        label = torch.zeros(self.S, self.S, 5 + self.C)
        bnds = o['annotation']['object']
        xscale = self.image_size / int(o['annotation']['size']['width'])
        yscale = self.image_size / int(o['annotation']['size']['height'])

        for bnd in bnds:
            name = classes.index(bnd["name"])
            one_hot = F.one_hot(torch.tensor(name), self.C)
            b = bnd['bndbox']
            xmin = (int(b['xmin']) * xscale) / self.image_size
            ymin = (int(b['ymin']) * yscale) / self.image_size
            width = ((int(b['xmax']) - int(b['xmin'])) * xscale) / self.image_size # width and height are relative to whole image 
            height = ((int(b['ymax']) - int(b['ymin'])) * yscale) / self.image_size # so we done have to scale them

            xcenter = xmin + width / 2
            ycenter = ymin + height / 2

            # now need to get [x, y, w, h] in terms of specific cell center is in (i, j)

            i, j = int(xcenter * self.S), int(ycenter * self.S)

            # also only add this class if its only one in the cell cuz yolov1 sucks
            if label[i, j, 4] == 0: # check if cell occupies object
                label[i, j] = torch.cat([torch.tensor([xcenter, ycenter, width, height, 1]), one_hot]) # must keep shape

        return self.data[idx][0], label


def get_dataset(batch_size=16, S=13, B=5, C=20, image_size=416):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(0.4, 0.4, 0.4, hue=0.02),
        transforms.ToTensor(),
    ])
    data = torchvision.datasets.VOCDetection("data", '2012', 'train', download=False, transform=transform)
    dataloader = DataLoader(YoloDataset(data, S, B, C, image_size), batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
    return dataloader

