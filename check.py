import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from data import VOC_CLASSES as labels
# from experiments.ssdv8 import build_ssd
from ssd import build_ssd

import xml.etree.ElementTree as ET 
import math 
import time
from tqdm import tqdm
from thop import profile
from torchstat import stat


def generate_sub_feat(x,x_origin):
    '''冠军方案改进版  map=0.87,但是效果比方案1好，几乎都有1.00'''
    img_a_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    img_b_gray = cv2.cvtColor(x_origin, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('./img_a_gray.jpg',img_a_gray)
    # cv2.imwrite('./img_b_gray.jpg',img_b_gray)
    defect_diff=cv2.absdiff(x,x_origin)
    # cv2.imwrite('./defect_diff.jpg',defect_diff)
    origin_diff=cv2.bitwise_xor(defect_diff,x_origin)
    # cv2.imwrite('./origin_diff1.jpg',origin_diff)
    origin_diff=cv2.absdiff(origin_diff,x)
    # cv2.imwrite('./origin_diff2.jpg',origin_diff)
    defect_diff=cv2.cvtColor(defect_diff,cv2.COLOR_BGR2GRAY)
    origin_diff=cv2.cvtColor(origin_diff,cv2.COLOR_BGR2GRAY)
    defect_diff=cv2.merge([img_a_gray,defect_diff*3,img_b_gray-defect_diff])
    origin_diff=cv2.merge([img_a_gray,origin_diff*3,img_b_gray])
    return defect_diff,origin_diff

def check_speed(net):
	image_path1 = './test/0002_test.jpg'
	image_path2 = './test/0002_temp.jpg'
	image1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
	image2 = cv2.imread(image_path2, cv2.IMREAD_COLOR) 
	start=time.time()
	defect_diff,origin_diff = generate_sub_feat(x=image1,x_origin=image2)
	end=time.time()
	print(1000.*(end-start)," ms") 
	pre_time=1000.*(end-start)
	rgb_image1 = cv2.cvtColor(defect_diff, cv2.COLOR_BGR2RGB)
	rgb_image2 = cv2.cvtColor(origin_diff, cv2.COLOR_BGR2RGB)
	x = cv2.resize(rgb_image1, (model_input, model_input)).astype(np.float32)
	x -= (104.0, 117.0, 123.0)
	x = x.astype(np.float32)
	x = x[:, :, ::-1].copy()
	x = torch.from_numpy(x).permute(2, 0, 1)

	xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
	if torch.cuda.is_available():
		xx = xx.cuda()

	yy = cv2.resize(rgb_image2, (model_input, model_input)).astype(np.float32)
	yy -= (104.0, 117.0, 123.0)
	yy = yy.astype(np.float32)
	yy = yy[:, :, ::-1].copy()
	yy = torch.from_numpy(yy).permute(2, 0, 1)

	yyy = Variable(yy.unsqueeze(0))     # wrap tensor in Variable
	if torch.cuda.is_available():
		yyy = yyy.cuda()
	start1=time.time()
	y = net(xx,yyy)
	top_k=10
	detections,_,_ = y 
	end1=time.time()
	test_time=(end1-start1)*1000
	return test_time,pre_time

def check_Flop(net):
	stat(net,(3,300,300))
	# print("%.2fM"%(flop/1e6),"%.2fM"%(para/1e6))

# weight_path = './weights/baseline+fdclossvoc.pth'
model_input = 300

net = build_ssd('test', model_input,2)  
net = net.cpu()
# net = net.cuda()
# summary(net,(3,300,300))
# net.load_weights(weight_path)
# times=0
# pre_T=0
# for i in tqdm(range(5)):
# 	test_time,pre_time=check_speed(net)
# 	times+=test_time
# 	pre_T+=pre_time
	
# print("检测时间：",times/5) # 算法推理时间 4679.47ms
# print("预处理算法检测时间：",pre_T/5) # 24.41ms

check_Flop(net)
# total=sum([param.nelement() for param in net.parameters()])
# print("Number of parameter: %2fM"%(total/1e6))
# inputs=torch.randn(1,3,300,300)
# inputs=inputs.cuda()
# output,_=net(inputs)



# Number of parameter: 41.195636M

