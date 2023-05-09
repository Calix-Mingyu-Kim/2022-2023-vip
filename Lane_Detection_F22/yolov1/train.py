import argparse

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from model import YOLOv1
from bdd100k import BDD100k
from utils import non_max_supression, mean_average_precission, intersection_over_union
from loss import YoloLoss

import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

writer = SummaryWriter()

device = torch.device('cuda')
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

seed = 123
torch.manual_seed(seed)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--root', type=str, default='/home/pumetu/Purdue/LaneDetection/BDD100k/', help='root directory for both image and labels')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    return parser.parse_args()

def train(data_loader, model, optimizer, yoloLoss):
    '''
    Training loop of the model
    Args:
        data_loader: data loader for the dataset
        model: model to be trained
        optimizer: optimizer function we are using (SGD or ADAM or anything else)
        yoloLoss: loss function
    Return:
        running_loss: loss for the epoch (total)
    '''
    model.train()
    running_loss = 0
    for batch_idx, (imgs, targets) in enumerate(data_loader):
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss = yoloLoss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    return running_loss

def main():
    args = parse_arg()

    #Load YOLO model
    model = YOLOv1(split_grids=7, num_bboxes=2, num_classes=13).to(device)

    #Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    yoloLoss = YoloLoss()

    #Resize image to be shape 448 x 448 as indicated by the paper
    transform = transforms.Compose([
        transforms.Resize((448,448))])

    #Load BDD100k Dataset
    train_dataset = BDD100k(root='/home/pumetu/Purdue/LaneDetection/BDD100k/', train=True, transform=transform)
    val_dataset = BDD100k(root='/home/pumetu/Purdue/LaneDetection/BDD100k/', train=False, transform=transform)

    train_loader = data.DataLoader(dataset=train_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle=False)

    #Training Loop
    print("start training")
    losses = []
    for epoch in range(args.epoch):
        epoch_loss = train(train_loader, model, optimizer, yoloLoss)
        epoch_loss /= len(train_loader)
        print(f'Epoch {epoch}')
        print(f'{epoch_loss=}')
    
    #Save Model 
    torch.save(model.state_dict(), 'weights/yolov1_50.pt')

    #Plot
    plot_loss_curve()

if __name__ == '__main__':
    main()