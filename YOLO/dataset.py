import glob

import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data

# from ..utils.utils import xyxy_to_xywh

class BDD100k(data.DataLoader):
    def __init__(self, root, train=True, transform=None, S=7):
        self.root = root 
        self.train = train
        self.transform = transform

        self.detect = pd.read_json(self.root + 'labels/det20/det_val.json')
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
        }

        self.S = S
        self.C = len(self.class_dict)

    def __len__(self):
        return len(self.detect.index)

    def __getitem__(self, index):
        # if torch.is_tensor(index):    #convert tensor index to list
        #     index = index.tolist()
        img_row = self.detect.iloc[index]
        img_name = img_row['name']
        image = Image.open(img_name)
        
        
        labels = img_row['labels']
        bboxes = []
        fixed_w, fixed_h = 1280, 720
        
        for label in labels:
            # Assign x1, y1, x2, y2 and convert to bounding boxes of x, y, w, h
            box = label['box2d']
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            x, y, w, h = (x1 + x2) / (2 * fixed_w), (y1 + y2) / (2 * fixed_h), x2 - x1, y2 - y1
            category = self.class_dict[label['category']]
            bbox = [x, y, w, h, category]
            bboxes.append(bbox)
            
        # 7 x 7 x 20 image
        label = torch.zeros(self.S, self.S, self.C + 5) #(7x7x15)
        

        for box in bboxes: 
            # 1020, 300 / 1280, 720 = 0.79, 0.417
            # Find ith, jth grid
            i = int(box[0] * self.S)
            j = int(box[1] * self.S)

            # Put information in equivalent indices
            label[i, j, 4 + box[4]] = 1   # for equivalent object, add 4 because self.class_dict starts from 1
            
            # Replace with equivalent position
            label[i, j, :5] = box
            
        return image, label
        
if __name__ == "__main__":
    dataset = BDD100k('/Users/calixkim/VIP27920/bdd100k')
    dataloader = data.DataLoader(dataset)
        
    
    '''
    How to load in 1 image with its corresponding label

    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

    Don't have to do transform i.e. (data augmentation)
    
            1 image
    the label for that image
    how to find the bbox is in the center of that cell
    x1, y1, x2, y2
    (x1, y1) -> top left
                    (x2, y2) -> bottom right
    x, y, w, h
            (x, y)->center of box ->>>width
            height
    which grid cell does this x, y fall into
    1. Grid cell is responsible for predicting bbox
        -> the center of the bbox is in that grid cell
    2. The bounding box is responsible
        -> highest IOU
    (confidence, x, y, w, h) 5 * B
    groundtruth (confidence, x, y, w, h)
        confidence = 1 when the grid cell is responsible for detecting the object
        else confidence = 0
    return image, label -> 7, 7, 25
    '''