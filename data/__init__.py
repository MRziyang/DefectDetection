from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT

from .coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, get_label_map
from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    imgs_origin=[]

    for sample in batch:
        imgs.append(sample[0])
        imgs_origin.append(sample[1])
        targets.append(torch.FloatTensor(sample[2]))
    return torch.stack(imgs, 0), torch.stack(imgs_origin, 0),targets


def base_transform(image,size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image,image_origin, boxes=None, labels=None,aux_target=None):
        return base_transform(image, self.size, self.mean), \
               base_transform(image_origin, self.size, self.mean),boxes, labels
