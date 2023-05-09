import glob

import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision.io import read_image

from utils import xyxy_to_xywh

class BDD100k(data.DataLoader):
    def __init__(self, root, train=True, transform=None, S=7):
        self.root = root 
        self.train = train
        self.transform = transform

        self.detect = pd.read_json(self.root + 'labels/det_20/det_train.json') if train else pd.read_json(self.root + 'labels/det_20/det_val.json')
        self.detect.dropna(axis=0, subset=['labels'], inplace=True)

        self.lane = glob.glob(self.root + 'labels/lane/*.png')
        self.drivable = glob.glob(self.root + 'labels/drivable/*.png')

        self.class_dict = {
            'pedestrian' : 1,
            'rider' : 2,
            'car' : 3,
            'truck' : 4, 
            'bus' : 5, 
            'train' : 6, 
            'motorcycle' : 7,
            'bicycle' : 8,
            'traffic light' : 9,
            'traffic sign' : 10,
            'other vehicle': 11,
            'trailer': 12,
            'other person': 13,
        }

        self.S = S
        self.C = len(self.class_dict)
        
    def __len__(self):
        return len(self.detect.index)
  
    def __getitem__(self, index):
        target = self.detect.iloc[index]

        img_path = self.root + 'images/100k/train/' if self.train else self.root + 'images/100k/val/'
        img = read_image(img_path + target['name']) 
        img = img.type(torch.float32)
        _, height, width = img.shape
        
        annotations = target['labels']
        bboxes = []
        if not isinstance(annotations, list):
            print(target)
            
        for obj in annotations:
            obj_class = self.class_dict[obj['category']]
            bbox = list(obj['box2d'].values())
            bbox = xyxy_to_xywh(bbox, width, height, norm=True) 
            box_tensor = torch.Tensor(([obj_class] + bbox.tolist()))
            bboxes.append(box_tensor)
    
        if self.transform:
            img = self.transform(img)

        label = torch.zeros((self.S, self.S, self.C + 5))
        cell_size = 1. / self.S

        for bbox in bboxes:
            obj_class, x, y, w, h = bbox.tolist()
            obj_class = int(obj_class)
            i, j = int(self.S * y), int(self.S * x)
            x_, y_ = self.S * x - j, self.S * y - i 
            w_, h_ = self.S * w, self.S * h
            bbox_ = torch.tensor([x_, y_, w_, h_])
        
            if label[i, j, self.C] == 0:
                label[i, j, self.C] = 1
                label[i, j, self.C+1:self.C+5] = bbox_
                label[i, j, obj_class] = 1

        return img, label