import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
import torchvision.models as models
#from deformConv.deform_module import DeformConvNet
#from deformConv.layers import ConvOffset2D
from deformConvModule.deform_net import DeformConvNet
from layers.modules.switchable_norm import SwitchNorm2d
from deformConvModule.seLayer import SELayer

class SSD(nn.Module):

    def __init__(self, phase, size, base,Decoder_block,extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc['SSD{}'.format(size)]
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.base = nn.ModuleList(base) #resnet50 backbone

        self.Decoder_Block=nn.ModuleList(Decoder_block)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(2048, 20) #2048
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, size, 0, 200, 0.01, 0.7)

        # self.fc_feat=nn.Conv2d(4096,2048,kernel_size=1)
        # self.upsample=nn.UpsamplingBilinear2d(300)
        # self.avgpool=nn.AdaptiveAvgPool2d(1)
        # self.dropout=nn.Dropout(0.5)

    def forward_once(self,x):
        sources = list()
        res_sources = list()
        loc = list()
        conf = list()

        for k in range(len(self.base)):
            x = self.base[k](x)
            res_sources.append(x)

        s = self.L2Norm(x)  # 2048,1,1
        mix_feat, sources = Decode_feature(self.Decoder_Block, s, res_sources, sources)

        x = mix_feat  # 这里需要还原回1024,19,19

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                # sources.append(x)
                # def_x = DeformConvNet(x.shape[1])(x)  # 可变形卷积
                x = self.base[k](x)
                


        # apply multibox head to source layers 此处针对整个图片来操作
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output,sources[2]
    def forward(self, x,x_origin):
        x_output,feat_x=self.forward_once(x)
        _,feat_x_origin=self.forward_once(x_origin)
        return x_output,feat_x,feat_x_origin

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Begin loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage),False)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')





# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def resnet50(cfg, i = 3, batch_norm=False):
    layers = []
    model=models.resnet50(False)
    model.load_state_dict(torch.load('weights/resnet50-19c8e357.pth'))

    layers += [nn.Sequential(*list(model.children())[:-4])] #512
    layers += [nn.Sequential(*list(model.children())[-4:-3])] #1024
    layers += [nn.Sequential(*list(model.children())[-3:-2])] #2048
    print('resnet50 base:',layers) #output 512
    return layers


def Decoder_block(cfg,i,batch_norm=False):
   in_channels=i
   layers=[]
   layers+=[
       nn.UpsamplingBilinear2d(10),
        #2048--1024
       nn.Conv2d(in_channels,in_channels//2,1), #0
       nn.BatchNorm2d(in_channels//2),

       nn.ConvTranspose2d(in_channels//2, in_channels//2, 3, stride=2, padding=1),
       nn.ReLU(),

       #1024--512
       nn.Conv2d(in_channels//2,in_channels//4,1), 
       nn.BatchNorm2d(in_channels//4),
       nn.ReLU(),

       nn.ConvTranspose2d(in_channels//4,in_channels//4,3,stride=2,padding=1,output_padding=1),
       nn.ReLU(),
   ]
   return layers 

#2 switch module
# def Decoder_block(cfg,i,batch_norm=False):
#     in_channels=i
#     layers=[]
#     layers+=[
#         nn.Conv2d(in_channels,in_channels,3,1,1),#smooth 1
#         # nn.UpsamplingBilinear2d(10),
#         #2048--1024
#         nn.Conv2d(in_channels,in_channels//2,1),
#         SwitchNorm2d(in_channels//2),

#         nn.ConvTranspose2d(in_channels//2, in_channels//2, 3, stride=2, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(in_channels//2,in_channels//2,3,1,1),#smooth 2
#         #1024--512
#         nn.Conv2d(in_channels//2,in_channels//4,1),
#         SwitchNorm2d(in_channels//4),
#         nn.ReLU(),

#         nn.ConvTranspose2d(in_channels//4,in_channels//4,3,stride=2,padding=1,output_padding=1),
#         nn.ReLU(),
#         nn.Conv2d(in_channels//4,in_channels//4,3,1,1),#smooth 3
#     ]
#     print(layers)
#     return layers



def Decode_feature(block,x,before_feat,sourceList):
    '''还原回38*38'''
    x=x+before_feat[2]

    for i in range(0,5):
        x=block[i](x)
    x=x+before_feat[1]
    x_1024_f=x 
    
    for i in range(5,9):
        x=block[i](x)
    x=x+before_feat[0]
    x_512_f=x
    sourceList.append(x_512_f)
    sourceList.append(x_1024_f)

    return x_1024_f,sourceList
def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [
                    nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if len(cfg) == 13:
        print('input channels:',in_channels)
        layers += [nn.Conv2d(in_channels, 256, kernel_size=4,padding=1)]      # Fix padding to match Caffe version (pad=1).
    print('extras layers:',layers)
    return layers


def multibox(vgg,Decoder_block,extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    print('extra layer size:', len(extra_layers))
    for i, layer in enumerate(extra_layers):
        print('extra layer {} : {}'.format(i, layer))
    
    loc_layers += [nn.Conv2d(512,cfg[0] * 4, kernel_size=3, padding=1),
                   nn.Conv2d(1024,cfg[1] * 4, kernel_size=3, padding=1), ] #256
    conf_layers += [nn.Conv2d(512,cfg[0] * num_classes, kernel_size=3, padding=1),
                    nn.Conv2d(1024,cfg[1] * num_classes, kernel_size=3, padding=1),] #512

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]

    return vgg,Decoder_block,extra_layers, (loc_layers, conf_layers)

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 4, 4, 4],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size not in [300, 512] :
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 and SSD512 is supported!")
        return

    base_,Decoder_block_,extras_, head_ = multibox(resnet50(base[str(size)], 3),
                                     Decoder_block(base[str(size)],2048),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    print('Begin to build SSD-VGG...\n')
    return SSD(phase, size, base_,Decoder_block_, extras_, head_, num_classes)
