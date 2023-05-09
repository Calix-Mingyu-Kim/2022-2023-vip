import torch
import torch.nn as nn

from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, split_grids=7, num_bboxes=2, num_classes=13, lambda_coord=5, lambda_noobj=0.5):
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
        Returns:
            loss (tensor): loss value
        '''
        mse = nn.MSELoss(reduction='sum')
        Iobj = target[...,self.C].unsqueeze(-1)
        eps = 1e-6
        iou = []
        for i in range(self.B):
            iou_ = intersection_over_union(preds[..., self.C + 1 + 5 * i:self.C + 5 * (i+1)], target[...,self.C + 1:self.C + 5])
            iou.append(iou_)
        iou = torch.stack(iou)

        iou_b1 = intersection_over_union(preds[..., 14:18], target[..., 14:18])
        iou_b2 = intersection_over_union(preds[..., 19:23], target[..., 14:18])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        _, resbox = torch.max(iou, dim=0)

        bbox_preds = Iobj * ((resbox * preds[..., 19:23]) + ((1 - resbox) * preds[..., 14:18]))
        bbox_target = Iobj * target[..., 14:18]

        bbox_preds[..., 2:4] = torch.sign(bbox_preds[...,2:4]) * torch.sqrt(torch.abs(bbox_preds[..., 2:4] + eps)) 
        bbox_target[..., 2:4] = torch.sqrt(bbox_target[...,2:4]) 

        bbox_loss = mse(torch.flatten(bbox_preds, end_dim=-2), torch.flatten(bbox_target, end_dim=-2))

        confidence = resbox * preds[...,17:18] + (1 - resbox) * preds[...,13:14]
        conf_loss = mse(torch.flatten(Iobj * confidence), torch.flatten(Iobj * target[..., 13:14]))

        no_obj_loss = mse(torch.flatten((1 - Iobj) * preds[...,13:14], start_dim=1), torch.flatten((1 - Iobj) * target[...,13:14], start_dim=1))
        no_obj_loss += mse(torch.flatten((1 - Iobj) * preds[...,18:19], start_dim=1), torch.flatten((1 - Iobj) * target[...,13:14], start_dim=1))

        class_loss = mse(torch.flatten(Iobj * preds[...,:13], end_dim=-2), torch.flatten(Iobj * target[...,:13], end_dim=-2))
        
        loss = self.lambda_coord * bbox_loss + conf_loss + self.lambda_noobj * no_obj_loss + class_loss

        return loss