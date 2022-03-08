from __future__ import absolute_import, division

import torch
import torch.nn.functional as F
import torch.nn as nn
from deformConvModule.layers import ConvOffset2D


def make_layer(in_channels):
	layers=[]
	layers += [
		# nn.UpsamplingBilinear2d(10),
		# 2048--1024
		nn.Conv2d(in_channels, in_channels, 3, padding=1),
		nn.ReLU(),
		# nn.BatchNorm2d(in_channels),

		ConvOffset2D(in_channels),
		nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, stride=2),
		nn.ReLU(),
		nn.Dropout(p=0.5),
		# nn.BatchNorm2d(in_channels * 2),

		ConvOffset2D(in_channels * 2),
		nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, stride=2),
		nn.ReLU(),
		nn.Dropout(p=0.5),
		# nn.BatchNorm2d(in_channels),

		ConvOffset2D(in_channels),
		nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=2),
		nn.ReLU(),
		nn.Dropout(p=0.5),
		# nn.BatchNorm2d(in_channels),
	]
	return  layers
class DeformConvNet(nn.Module):
	def __init__(self,in_channels):
		super(DeformConvNet, self).__init__()
		self.layers=nn.ModuleList(make_layer(in_channels))
		# # conv11
		# self.conv11 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
		# self.bn11 = nn.BatchNorm2d(in_channels)
		#
		# self.convOffset1=ConvOffset2D(in_channels)
		# self.conv22=nn.Conv2d(in_channels,in_channels*2, 3, padding=1, stride=2)
		# self.bn22=nn.BatchNorm2d(in_channels*2)
		#
		# self.convOffset2=ConvOffset2D(in_channels*2)
		# self.conv33=nn.Conv2d(in_channels*2,in_channels, 3, padding=1, stride=2)
		# self.bn33=nn.BatchNorm2d(in_channels)
		#
		# self.convOffset3=ConvOffset2D(in_channels)
		# self.conv44=nn.Conv2d(in_channels,in_channels, 3, padding=1, stride=2)
		# self.bn44=nn.BatchNorm2d(in_channels)
	

	def forward(self, x):
		for i in range(len(self.layers)):
			x=self.layers[i](x)
		# x = F.relu(self.conv11(x))
		# x = self.bn11(x)
		# x=self.convOffset1(x)
		# x=self.conv22(x)
		# x=self.bn22(F.relu(x))
		# x = self.convOffset2(x)
		# x = self.conv33(x)
		# x = self.bn33(F.relu(x))
		# x = self.convOffset3(x)
		# x = self.conv44(x)
		# x = F.relu(x)

		x=F.avg_pool2d(x,kernel_size=[x.size(2),x.size(3)])
		
		return x