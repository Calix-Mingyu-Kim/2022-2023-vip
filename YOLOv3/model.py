import torch
import torch.nn as nn

'''
Darknet-53 Implementation
Author: Mingyu Kim
'''
class ConvBlock(nn.Module):
    '''
    class ConvBlock constructs each convolution layers including nn.Conv2d, nn.BatchNorm2d, and nnLeakyReLU.
    Forward pass: Convolution -> Batchnorm -> LeakyReLU
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
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
            padding (int): padding of input. 
        '''
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        '''
        Forward pass: Convolution -> Batchnorm -> LeakyReLU
        '''
        x = self.relu(self.bn(self.conv(x)))

        return x

class ResBlock(nn.Module):
    '''
    class Resblock constructs each Residual blocks. In darknet-53 all parameters 
    remain the same except the in_channel gets multiplied by two in each iteration.
    Forward pass: conv1x1, conv3x3. output: F(x) + x
    input and output size remains the same. 
    '''
    def __init__(self, in_channels):
        '''
        Args:
            in_channels (int): number of channels in input image or feature map
        '''
        super(ResBlock, self).__init__()
        self.conv1x1 = ConvBlock(in_channels, in_channels//2, 1, 1, 0)
        self.conv3x3 = ConvBlock(in_channels//2, in_channels, 3, 1, 1)
    
    def forward(self, x):
        '''
        Forward pass: conv 1x1, conv 3x3
        output = H(x) = F(x) + x
        Maintain input & output size
        '''
        out = self.conv1x1(x)
        out = self.conv3x3(out)
        
        return x + out  # simple addition

class DarkNet53(nn.Module):
    '''
    class Darknet53 is the entireity of darknet-53. 
    Forward pass: _build_conv, build the entire architecture and output torch.size([# of images, N, N, # of channels])
    '''
    def __init__(self, in_channels=3, num_block=[1, 2, 8, 8, 4], init_block_filter=64):
        '''
        YOLOv3 implementation based on the paper YOLOv3: An Incremental Improvement (https://arxiv.org/pdf/1804.02767v1.pdf)
        Args:
            in_channels (int): number of channels of the input image
            num_block (list): number of repeated convolutional block (e.g. conv block in idx 2 will repeat 8 times.)
            init_block_filter (int): initial number of filters of 3 x 3 / 2 convolutional layers; multiplied by two (up to 1024) in every convolutional block 
        '''
        super(DarkNet53, self).__init__()
        self.num_block = num_block
        self.init_block_filter = init_block_filter
        self.top_layer = ConvBlock(in_channels, 32, 3, 1)
        self.conv = self._build_conv()

    def forward(self, x):
        '''
        Forward pass: Convolutional Layers (Feature Extraction) -> Fully Connected Layers
        Final output is: torch.size([2, 13, 13, 1024]) when N = 13. 
        '''
        x = self.conv(x)
        
        return x.view(2, 13, 13, -1)

    def _build_conv(self):
        '''
        Builds the convolutional layers for feature extraction based on config
        O = ((W - F + 2P) / S) + 1 
        O: output size, W: input size, F: size of the kernel, P: padding, S: stride.
        '''
        model = []
        model.append(self.top_layer)
        in_channels = self.init_block_filter
        for res_block in self.num_block:
            model.append(ConvBlock(in_channels // 2, out_channels=in_channels, kernel_size=3, stride=2))
            for _ in range(res_block):
                model.append(ResBlock(in_channels=in_channels))
            in_channels *= 2
        
        return nn.Sequential(*model)

# model should get (13, 13, 1024)
model = DarkNet53()
x = torch.randn((2, 3, 416, 416))
print(model(x).shape)