import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, split_grids=7, num_bboxes=2, num_classes=10, lambda_coord=5, lambda_noobj=0.5):
        '''
        YOLO loss function
        Args:
            split_grids (int): number of grids to split image into
            num_bboxes (int): number of bounding boxes to predict per grid
            num_classes (int): number of classes in dataset
            lambda_coord (int): weight for bounding box location loss
            lambda_noobj (int): weight for not having an object in the grid cell loss
        '''
        super().__init__()
        self.S = split_grids
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self, preds, target):
        '''
        4 Parts to YOLO Loss Function:
            1. Loss related to (x,y) of bounding box
            2. Loss related to (w,h) of bounding box
            3. Loss related to confidence score for each bounding box
            4. Classification loss
        Args:
            preds (tensor): Prediction from YOLO, size: (N, S, S, (B * 5) + C)
            target (tensor): Groundtruth, size : (N, S, S, 5 + C)
        '''
        total_loss = torch.zeros(1)
        # xy loss
        loss = nn.MSELoss(reduction='sum')
        # match the ground truth to whichever predicted box has the current highest IOU. compare bbox1 and bbox2
        # [x, y, w, h, C1, x, y, w, h, C2, 10 classes]
        if self.intersection_over_union(target[..., 0:5], preds[..., 0:5]) > self.intersection_over_union(target[..., 5:10], preds[..., 5:10]):
            idx = 0
        else:
            idx = 5
        
        # 1e^-6
        # [4, 7, 7, 1] 
        if len(target[..., :10]) > 1:   # if grid cell is responsible for predicting the bounding box: 
            total_loss += self.lambda_coord * loss(target[..., idx:idx+1], preds[..., idx:idx+1])   # x [..., 0:1|5:6]
            total_loss += self.lambda_coord * loss(target[..., idx+1:idx+2], preds[..., idx+1:idx+2])   # y [..., 1:2|6:7]
            total_loss += self.lambda_coord * loss(torch.sqrt(torch.abs(target[..., idx+2:idx+3])), torch.sqrt(torch.abs(preds[..., idx+2:idx+3]))) # w [..., 2:3|7:8]
            total_loss += self.lambda_coord * loss(torch.sqrt(torch.abs(target[..., idx+3:idx+4])), torch.sqrt(torch.abs(preds[..., idx+3:idx+4]))) # h [..., 3:4|8:9]
            total_loss += loss(target[..., 10:11], preds[..., 10:11])   # C
            total_loss += loss(target[..., 0:10], preds[0,10])  # classification
            
        else:   # if no object is present
            total_loss += self.lambda_noobj * loss(loss(target[..., idx+4:idx+5], preds[..., idx+4:idx+5]))  # [..., 4:5|9:10]
            
        return total_loss
    
    def intersection_over_union(self, box1, box2):
        #box1[0] -> x
        #box1[2] -> w
        x_min_1 = torch.clamp(box1[0] - box1[2] / 2, 0, 1) # x - w/2    # min is 0, max is 1
        x_max_1 = torch.clamp(box1[0] + box1[2] / 2, 0, 1) # x + w/2
        y_min_1 = torch.clamp(box1[1] - box1[3] / 2, 0, 1) # y - w/2
        y_max_1 = torch.clamp(box1[1] + box1[3] / 2, 0, 1) # y + w/2
        
        x_min_2 = torch.clamp(box2[0] - box2[2] / 2, 0, 1) # x - w/2
        x_max_2 = torch.clamp(box2[0] + box2[2] / 2, 0, 1) # x + w/2
        y_min_2 = torch.clamp(box2[1] - box2[3] / 2, 0, 1) # y - w/2
        y_max_2 = torch.clamp(box2[1] + box2[3] / 2, 0, 1) # y + w/2
        
        # coordinates of intersection
        x_l = torch.max(x_min_1, x_min_2)
        x_r = torch.min(x_max_1, x_max_2)
        y_b = torch.max(y_min_1, y_min_2)
        y_t = torch.min(y_max_1, y_max_2)
        
        intersection = (x_r - x_l) * (y_t - y_b) # area of intersection
        union = box1[2] * box1[3] + box2[2] * box2[3] - intersection # box1 w * h + box2 w * h - intersection
        
        iou = intersection / union
        
        return iou