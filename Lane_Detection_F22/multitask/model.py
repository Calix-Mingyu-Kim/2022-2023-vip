import torch
import torch.nn as nn

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

class ResBlock(nn.Module):
    def __init__(self, conv1x1, conv3x3):
        '''
        Initialize ResBlock object
        - Obtain parent class's information, functions, and fields using super().__init__() of nn.Module
        - Define variables and function calls using the object initialization parameters
        Args:
            conv1x1 (tuple): 
            conv3x3 (tuple): 
        '''
        super().__init__()

        in_ch1, out_ch1, filter1, padding1, s1 = conv1x1
        in_ch2, out_ch2, filter2, padding2, s2 = conv3x3
        self.firstConvBlock = ConvBlock(in_ch1, out_ch1, filter1, s1, padding1)
        self.secondConvBlock = ConvBlock(in_ch2, out_ch2, filter2, s2, padding2)

    def forward(self, x):
        '''
        Override of forward pass inherited from parent class nn.Module
        - Use function calls defined in object initialization to carry out desired forward pass
        '''
        input = x
        x = self.firstConvBlock(x)
        x = self.secondConvBlock(x)
        return x + input

class Darknet53(nn.Module):
    '''
    Darknet-53 Backbone
    '''
    def __init__(self):
        super().__init__()

        self.config = {
            'c1': [(3, 32, 3, 1, 1),
                   (32, 64, 3, 1, 2),
                   [(64, 32, 1, 0, 1), (32, 64, 3, 1, 1), 1]],
            'c2': [(64, 128, 3, 1, 2),
                   [(128, 64, 1, 0, 1), (64, 128, 3, 1, 1), 2]],
            'c3': [(128, 256, 3, 1, 2),
                   [(256, 128, 1, 0, 1), (128, 256, 3, 1, 1), 8]],
            'c4': [(256, 512, 3, 1, 2),
                   [(512, 256, 1, 0, 1), (256, 512, 3, 1, 1), 8]],
            'c5': [(512, 1024, 3, 1, 2),
                   [(1024, 512, 1, 0, 1), (512, 1024, 3, 1, 1), 4]],
        }

    def _build_model(self, conf):
        '''
        Builds nn.Sequential sequence of the model's layers
        - Using the config defined in the constructor, loop through the instructions
        - When instruction is a tuple, execute a single convolutional layer
        - When instruction is a list, execute a certain number of residual blocks (indicated in list)
        - Use information inside the tuple or list to determine the input channels, output channels,
        filter size, padding, and stride
        '''
        layers = []
        for x in conf:
            if type(x) == tuple:
                '''
                Execute convolutional layer from tuple as: (in_channels, out_channels, filter size, padding, stride)
                - Extract individual variables from tuple, then pass them into new ConvLayer object as parameters
                - Append this ConvLayer object to the layers list
                '''
                in_ch, out_ch, f_size, pad, s = x
                layers += [ConvBlock(in_ch, out_ch, f_size, s, pad)]
            elif type(x) == list:
                '''
                Execute a number (repeats) of residual blocks from list as: [ConvLayer tuple, ConvLayer tuple, repeats]
                - Where ConvLayer tuples are the convolutional layers to be passed into each residual block
                - Extract individual variables from list, then pass them into new ResBlock object as parameters
                - Use a for loop executing "repeats" number of times to append a number of residual blocks to the sequence
                - For each residual block, use the conv1x1 and conv3x3 tuples to pass into a new ResBlock object and add to the list
                '''
                conv1x1, conv3x3, repeats = x[0], x[1], x[2]
                for _ in range(repeats):
                    layers += [ResBlock(conv1x1, conv3x3)]
        
        return nn.Sequential(*layers)
    
    def forward(self, x):

        c1 = self._build_model(self.config['c1'])(x)
        c2 = self._build_model(self.config['c2'])(c1)
        c3 = self._build_model(self.config['c3'])(c2)
        c4 = self._build_model(self.config['c4'])(c3)
        c5 = self._build_model(self.config['c5'])(c4)

        return c1, c2, c3, c4, c5

class Neck(nn.Module):
    '''
    Neck of the network comprises of FPN and SPP
    '''
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

class YoloHead(nn.Module):
    '''
    Detection Head
    Args:
        p {key: str, values: tensor}: layers p3 - p5  
    '''
    def __init__(self):
        super().__init__()
        
    def conv1x1(self, x):
        return (ConvBlock(x.shape[1]), 90, 1, 1, 0)
        
    # Desired output: S x S x 90 (5*(5+13))
    def forward(self, p3, p4, p5):
        p3 = self.conv1x1(p3)
        p4 = self.conv1x1(p4)
        p5 = self.conv1x1(p5)
        
        return p3, p4, p5

class LaneHead(nn.Module):
    '''
    Segmentation Head (UNet Like Structure)
    '''
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class YoloMulti(nn.Module):
    '''
    Multi-Task network based on YOLO framework
    '''
    def __init__(self, n_classes=13):
        super().__init__()
        output_channels = (5 + n_classes) * 5 

        #Backbone   
        self.darknet53 = Darknet53()
        # #Neck
        # self.neck = Neck()
        # #Heads
        self.detect_head = YoloHead()
        # self.lane_head = LaneHead()

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.darknet53(x)
        # p3, p4, p5 = self.neck(c3, c4, c5)
        # detect = self.detect_head(p3, p4, p5)
        # lane = self.lane_head(c1, c2, p3, p4, p5)
        print(c1.shape)
        print(c2.shape)
        print(c3.shape)
        print(c4.shape)
        print(c5.shape)

        return c1

x = torch.randn(2, 3, 416, 416)
model = YoloMulti()
out = model(x)
