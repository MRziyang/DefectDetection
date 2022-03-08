"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# VOC_CLASSES = ('double','broke','dirty','miss')
VOC_CLASSES=('0','1')
# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join('./', "data/VOCdevkit/")


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(eval(bbox.find(pt).text)) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):

    def __init__(self, root,
                 image_sets=[('data1.0', 'train')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='data1.0'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.ids_origin=list() #模板图
        for (year, name) in image_sets:
            rootpath = osp.join(self.root,  year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

                line_origin=line.replace('test','temp') #把模板引入
                self.ids_origin.append((rootpath, line_origin.strip()))

    def __getitem__(self, index):
        im,im_ori, gt, h, w = self.pull_item(index)

        return im,im_ori, gt

    def __len__(self):
        return len(self.ids)


    # def generate_sub_feat(self,x,x_origin):

    #     #读入原图
    #     img_a = cv2.cvtColor(x, cv2.IMREAD_COLOR)
    #     img_b = cv2.cvtColor(x_origin, cv2.IMREAD_COLOR)
    #     return img_b,img_a

    def generate_sub_feat(self,x,x_origin):
        '''冠军方案改进版  map=0.87,但是效果比方案1好，几乎都有1.00'''
        img_a_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        img_b_gray = cv2.cvtColor(x_origin, cv2.COLOR_BGR2GRAY)

        defect_diff=cv2.absdiff(x,x_origin)
        origin_diff=cv2.absdiff(x_origin,defect_diff)
        origin_diff=cv2.absdiff(x,origin_diff)
        defect_diff=cv2.cvtColor(defect_diff,cv2.COLOR_BGR2GRAY)
        origin_diff=cv2.cvtColor(origin_diff,cv2.COLOR_BGR2GRAY)
        defect_diff=cv2.merge([img_a_gray,defect_diff*3,img_b_gray-defect_diff])
        origin_diff=cv2.merge([img_a_gray,origin_diff*3,img_b_gray])
        return defect_diff,origin_diff

    # def generate_sub_feat(self,x,x_origin):
    #     '''冠军方案改进版  map=0.87,但是效果比方案1好，几乎都有1.00'''
    #     img_a_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    #     img_b_gray = cv2.cvtColor(x_origin, cv2.COLOR_BGR2GRAY)

    #     defect_diff=cv2.absdiff(x,x_origin)
    #     origin_diff=cv2.absdiff(x_origin,defect_diff)
    #     origin_diff=cv2.absdiff(x,origin_diff)
    #     defect_diff=cv2.cvtColor(defect_diff,cv2.COLOR_BGR2GRAY)
    #     origin_diff=cv2.cvtColor(origin_diff,cv2.COLOR_BGR2GRAY)
    #     defect_diff=cv2.merge([img_a_gray,defect_diff*3,img_b_gray-defect_diff])
    #     origin_diff=cv2.merge([img_a_gray,origin_diff*3,img_b_gray])
    #     # diff=cv2.absdiff(x,x_origin) #缺陷区域
    #     # diff=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    #     # img=cv2.merge([img_a_gray,diff*3,img_b_gray-diff]) #改动的地方
    #     # img_origin=cv2.merge([img_a_gray,img_a_gray,img_a_gray])
    #     return defect_diff,origin_diff


    # def generate_sub_feat(self,x,x_origin):
    #     '''冠军方案改进版 map=90 这个会有0.86'''
    #     img_a_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    #     img_b_gray = cv2.cvtColor(x_origin, cv2.COLOR_BGR2GRAY)
    #
    #     defect_diff=cv2.absdiff(x,x_origin)
    #     origin_diff=cv2.absdiff(x_origin,defect_diff)
    #     defect_diff=cv2.cvtColor(defect_diff,cv2.COLOR_BGR2GRAY)
    #     origin_diff=cv2.cvtColor(origin_diff,cv2.COLOR_BGR2GRAY)
    #     defect_diff=cv2.merge([img_a_gray,defect_diff*3,img_b_gray-defect_diff])
    #     origin_diff=cv2.merge([img_a_gray,origin_diff*3,img_b_gray-origin_diff])
    #     # diff=cv2.absdiff(x,x_origin) #缺陷区域
    #     # diff=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    #     # img=cv2.merge([img_a_gray,diff*3,img_b_gray-diff]) #改动的地方
    #     # img_origin=cv2.merge([img_a_gray,img_a_gray,img_a_gray])
    #     return defect_diff,origin_diff

    def pull_item(self, index):
        img_id = self.ids[index]
        img_origin_id=self.ids_origin[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        img_origin = cv2.imread(self._imgpath % img_origin_id)

        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            # aux_target=np.array(aux_target)
            img,img_origin=self.generate_sub_feat(img,img_origin) #此处输入的是缺陷位置图和原图的缺陷前的位置图
            img,img_origin,boxes, labels = self.transform(img,img_origin,target[:, :4], target[:, 4])

            # to rgb
            img = img[:, :, (2, 1, 0)]
            img_origin = img_origin[:, :, (2, 1, 0)]

            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1),\
                torch.from_numpy(img_origin).permute(2, 0, 1), \
                    target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        img_id_origin = self.ids_origin[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR),cv2.imread(self._imgpath % img_id_origin, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)

        return img_id[1],gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
