import torch
import numpy as np
import matplotlib.pyplot as plt

def intersection_over_union(preds, targets):
    '''
    Calculates the intersection over union for two bounding boxes of format x, y, w, h
    IOU = intersection / union
    Args:
        preds (tensor): predicted bounding box, size : [N, 4]
        targets (tensor): groundtruth bounding box, size : [M, 4]
    '''
    box1_x1 = preds[..., 0:1] - preds[..., 2:3] / 2
    box1_y1 = preds[..., 1:2] - preds[..., 3:4] / 2
    box1_x2 = preds[..., 0:1] + preds[..., 2:3] / 2
    box1_y2 = preds[..., 1:2] + preds[..., 3:4] / 2
    box2_x1 = targets[..., 0:1] - targets[..., 2:3] / 2
    box2_y1 = targets[..., 1:2] - targets[..., 3:4] / 2
    box2_x2 = targets[..., 0:1] + targets[..., 2:3] / 2
    box2_y2 = targets[..., 1:2] + targets[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)


def non_max_supression(bboxes, iou_thresh, thresh):
    '''
    Performs non max supression on the bounding box passed in
    '''
    bboxes = sorted([box for box in bboxes if box[1] > thresh], key = lambda x : -x[1])
    print("!!!", bboxes)
    nms = []

    while bboxes:
        box_ = bboxes.pop(0)

        bboxes = [box for box in bboxes if box[0] != box_[0] or intersection_over_union((box_[2:]).clone().detach(), (box[2:]).clone().detach()) < iou_thresh]        

        nms.append(box_)
    
    return nms
    

def xyxy_to_xywh(bbox, width, height, norm=False):
    '''
    Converts bounding box of format x1, y1, x2, y2 to x, y, w, h
    Args:
        bbox: bounding box with format x1, y1, x2, y2
        width: width of image
        height: height of image
        norm (bool): if True normalize the coordinates to the width and height of the image else send the exact pixel location
    Return:
        bbox_: bounding box with format x, y, w, h if norm is False else the coordinates are normalized to the height and width of the image
    '''
    bbox_ = np.copy(bbox)
    bbox_[0] = (bbox[0] + bbox[2]) / 2
    bbox_[1] = (bbox[1] + bbox[3]) / 2
    bbox_[2] = bbox[2] - bbox[0]
    bbox_[3] = bbox[3] - bbox[1]

    if norm:
        bbox_[0] /= width
        bbox_[1] /= height
        bbox_[2] /= width
        bbox_[3] /= height

    return bbox_

def xywh_to_xyxy(bbox, width, height):
    '''
    Converts bounding box of format x, y, w, h to x1, y1, x2, y2
    Args:
        bbox: bounding box with format x, y, w, h
        norm: was the bounding normalized
    Return:
        bbox_: bounding box with format x1, y2, x2, y2
    '''
    bbox_ = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
    bbox_[:, 0] = (bbox[:, 0] - bbox[:, 2] / 2) 
    bbox_[:, 1] = (bbox[:, 1] - bbox[:, 3] / 2)
    bbox_[:, 2] = (bbox[:, 0] + bbox[:, 2] / 2) 
    bbox_[:, 3] = (bbox[:, 1] + bbox[:, 3] / 2)

    return bbox_

def precision(tp ,fp):
    try:
        pre = tp / (tp + fp)
    except:
        pre = 1
    return pre
    
def recall(tp, fn):
    try:
        rec = tp / (tp + fn)
    except:
        rec = 1
    return rec

def f1_score():
    pass

def mean_average_precision(pred_boxes, true_boxes, iou_threshold = 0.5, num_classes = 13):
    avg_precisions = []
    epsilon = 1e-6
    
    for c in range(num_classes):
        detections = []
        ground_truths = []
        
        # for detection in pred_boxes:
            
    pass

def plot_loss_curve(epochs, losses):
    epochs = np.arange(1, epochs + 1)
    plt.plot(epochs, losses)