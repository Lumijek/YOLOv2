import torch
import torch.nn as nn
import torch.nn.functional as F
from time import perf_counter as time


class ConvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, alpha=0.1):
		super().__init__()
		if padding == None:
			padding = kernel_size // 2

		self.layer = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(alpha)
		)

	def forward(self, x):
		return self.layer(x)

class ReorgLayer(nn.Module):
    def __init__(self, s=2):
        super().__init__()
        self.s = s

    def forward(self, x):
        B, C, H, W = x.size()
        s = self.s
        x = x.view(B, C, H // s, s, W // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.contiguous().view(B, s * s * C, H // s, W // s)
        return x

class YOLOv2(nn.Module):
	def __init__(self, S=13, B=5, C=20):
		super().__init__()

		self.S = S
		self.B = B
		self.C = C
		# ConvNet
		self.conv1 = ConvLayer(3, 32, 3)
		self.mp1 = nn.MaxPool2d(2)

		self.conv2 = ConvLayer(32, 64, 3)
		self.mp2 = nn.MaxPool2d(2)

		self.conv3 = nn.Sequential(
			ConvLayer(64, 128, 3),
			ConvLayer(128, 64, 1),
			ConvLayer(64, 128, 3),
		)
		self.mp3 = nn.MaxPool2d(2)

		self.conv4 = nn.Sequential(
			ConvLayer(128, 256, 3),
			ConvLayer(256, 128, 1),
			ConvLayer(128, 256, 3),
		)
		self.mp4 = nn.MaxPool2d(2)

		self.conv5 = nn.Sequential(
			ConvLayer(256, 512, 3),
			ConvLayer(512, 256, 1),
			ConvLayer(256, 512, 3),
			ConvLayer(512, 256, 1),
			ConvLayer(256, 512, 3),
		)
		self.mp5 = nn.MaxPool2d(2)
		self.reorg = ReorgLayer(2)

		self.conv6 = nn.Sequential(
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 512, 1),
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 512, 1),
			ConvLayer(512, 1024, 3),
		)

		self.conv7 = nn.Sequential(
			ConvLayer(1024, 1024, 3),
			ConvLayer(1024, 1024, 3),
			ConvLayer(1024, 1024, 3)
		)

		self.output = ConvLayer(1024 + 2048, 125, 1)


	def forward(self, x):

		batch_size = x.shape[0]

		x = self.mp1(self.conv1(x))
		x = self.mp2(self.conv2(x))
		x = self.mp3(self.conv3(x))
		x = self.mp4(self.conv4(x))
		x = self.conv5(x)
		passthrough = self.reorg(x)
		x = self.mp5(x)
		x = self.conv6(x)
		x = self.conv7(x)
		x = torch.concat([passthrough, x], dim=1)
		x = self.output(x)
		print(x.shape)
		x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, self.S, self.S, self.B, 25)
		return x
