import torch
from torchvision.utils import draw_bounding_boxes

from utils import xywh_to_xyxy

def unnorm_bbox(bbox, width, height, i, j):
    '''
    Unormalize the bounding box predictions based on the yolo predictions
    '''
    bbox[:,0] = width / (7 / (bbox[:,0] + j))
    bbox[:,1] = height / (7 / (bbox[:,1] + i))
    bbox[:,2] = width / (7 / bbox[:,2])
    bbox[:,3] = height / (7 / bbox[:,3])

    return bbox

def draw_batch(img, pred, out_dir='./', display=False):
    '''
    Draw the predicted bounding boxes from the network on a single image
    Args:
        img (tensor): image of the predicted bboxes
        pred (tensor): predicted bboxes
        out_dir (str): output directory to store image
        display (bool): display the image or not
    Returns:
    '''
    class_dict = {
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
    num_to_class = {i:s for s,i in class_dict.items()}

    s = pred.shape[1]
    img = torch.tensor(img) if not isinstance(img, torch.Tensor) else img
    assert type(pred) == torch.Tensor
    img = img.to(torch.uint8)
    _, height, width = img.shape
    _, pclass = torch.max(pred[:, :, :13], dim=2)
    for i in range(s):
        for j in range(s):
            conf = pred[i, j, 13]
            if conf > 0.5:
                print(conf)
                class_pred = num_to_class[pclass[i, j].item()]
                label = [class_pred]
                bbox1 = pred[i, j, 14:18].unsqueeze(0)
                bbox1 = unnorm_bbox(bbox1, width, height, i ,j)
                bbox1 = xywh_to_xyxy(bbox1, width, height)
                bbox2 = pred[i, j, 19:23].unsqueeze(0)
                bbox2 = unnorm_bbox(bbox2, width, height, i ,j)
                bbox2 = xywh_to_xyxy(bbox2, width, height)
                img = draw_bounding_boxes(img, bbox1, width=3, labels=label, colors=(0, 255, 0)) 
                img = draw_bounding_boxes(img, bbox2, width=3, labels=label, colors=(0, 255, 0)) 

    return img

def draw_data_loader(imgs, labels):
    '''
    Draw the predicted bounding boxes from the network on a single image
    Args:
        imgs (tensor): images of the predicted bboxes
        labels (tensor): ground truth bboxes
    Returns:
        ground_truth (numpy array): ground truth bboxes with object in it. [batch #, x, y, w, h]
    '''
    b, _, height, width = imgs.shape
    imgs = imgs.to(torch.uint8)
    
    S = labels.shape[1]
    conf = labels[..., 13]
    I = torch.nonzero(conf).numpy()
    
    batch, i, j = I[:,0], I[:,1], I[:,2] 
    bbox = labels[batch,i,j,14:18]
    bbox[...,0] = width / (S / (bbox[...,0] + j))
    bbox[...,1] = height / (S / (bbox[...,1] + i))
    bbox[...,2] = width / (S / bbox[...,2])
    bbox[...,3] = height / (S / bbox[...,3])
    bbox = xywh_to_xyxy(bbox, width, height)
    batch = np.expand_dims(batch, axis=0)

    ground_truth = np.concatenate((batch.T,bbox),axis=1)
    
    return ground_truth