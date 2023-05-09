import torch
import torch.nn as nn

'''
YOLOv1 Architecture Config
Tuple: (out_channels, kernel_size, stride, padding)
String: "Maxpool" : maxpool layer kernel = (2 x 2) stride = 2
List: [(in_channels, out_channels, kernel_size, stride, padding), num_repeats]
'''
YOLOv1_config = [
    (64, 7, 2, 3),
    "Maxpool",
    (192, 3, 1, 1),
    "Maxpool",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "Maxpool", 
    [(256, 1, 1, 0), (512, 3, 1, 1), 4],
    (512, 1, 1, 0), 
    (1024, 3, 1, 1),
    "Maxpool",
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        '''
        Standard convolution block:
            1. Convolution
            2. Batch Normalization
            3. Activation Function (LeakyReLU)

        Args:
            in_channels (int): number of channels in input image or feature map
            out_channels (int): number of channels produced by the convolution
            kernel_size (int): size of convolving kernel
            stride (int): stride of convolution
            padding (int): padding of input
        '''
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        '''
        Forward pass: Convolution -> Batchnorm -> LeakReLU
        '''
        return self.act(self.bn(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, split_grids=7, num_bboxes=2, num_classes=13):
        '''
        YOLOv1 implementation based on the paper You Only Look Once: Unified, Real-Time Object Detection (https://arxiv.org/pdf/1506.02640.pdf)
        Args:
            in_channels (int): number of channels of input image
            split_grids (int): number of grids to split image into (S x S)
            num_boxes (int): number of boxes to predict per grid 
            num_class (int): number of classes in dataset
        '''
        super().__init__()
        self.config = YOLOv1_config
        self.in_channels = in_channels
        self.S = split_grids
        self.B = num_bboxes
        self.C = num_classes
        self.convs = self._build_conv()
        self.fcn = self._build_fcn()
    
    def forward(self, x):
        '''
        Forward pass: Convolutional Layers (Feature Extraction) -> Fully Connected Layers
        Final shape should be S x S x (B * 5 + C) 
            Hint: will need to do some sort of reshaping
        '''
        x = self.convs(x)
        x = self.fcn(x)
        return x.view(-1, self.S, self.S, (self.B * 5 + self.C))

    def _build_conv(self):
        '''
        Builds the convolutional layers for feature extraction based on config
        '''
        model = []
        in_channels = self.in_channels
        for layer in self.config:
            if isinstance(layer, tuple):
                model.append(ConvBlock(in_channels=in_channels, out_channels=layer[0], kernel_size=layer[1], stride=layer[2], padding=layer[3]))
                in_channels = layer[0]
            elif isinstance(layer, str):
                model.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(layer, list):
                conv1 = layer[0]
                conv2 = layer[1]
                for _ in range(layer[-1]):
                    model.append(ConvBlock(in_channels=in_channels, out_channels=conv1[0], kernel_size=conv1[1], stride=conv1[2], padding=conv1[3]))
                    model.append(ConvBlock(in_channels=conv1[0], out_channels=conv2[0], kernel_size=conv2[1], stride=conv2[2], padding=conv2[3]))
                    in_channels = conv2[0]
        
        return nn.Sequential(*model)


    def _build_fcn(self):
        '''
        Builds the fully connected layers       
        '''
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )
        
model = YOLOv1()
x = torch.randn((2, 3, 448, 448))
print(model(x).shape)