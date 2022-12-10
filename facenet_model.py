import torch
import torch.nn as nn


class FaceNet(nn.Module):
    """
    Model based on GoogLeNet (Inception v1) architecture and on facenet model architecture
    The first layers of GoogLeNet (Conv & Maxpool) have been removed as input image of our project
    is a 3x56x56 image.
        -> FaceNet paper : https://arxiv.org/abs/1503.03832
        -> GoogLeNet paper : https://arxiv.org/abs/1409.4842v1

    L2 pooling has been used in inception module as in facenet model.
    The final layer is a sigmoid (same for intermediate output modules).
    Input should be a (Nx3x56x56) tensor
    Output will be a (Nx1) tensor

    Args:
        droupout (float) : Dropout probability for final module prediction, 
                should be 0<dropout<1
        dropout_inter (float) : Dropout probability for intermediate modules 
                prediction, should be 0<dropout_inter<1
        init_weights (bool) : If true the weights are initialized with normal
                and constant distribution
    """
    def __init__(self, dropout:float, dropout_inter:float,init_weights:bool):


        super(FaceNet, self).__init__()
        self.conv1=ConvModule(3,3,1,1,0)
        self.conv2=ConvModule(3,64,1,1,0)
        self.conv3=ConvModule(64,192,3,1,1)
        self.maxpool1=nn.MaxPool2d(3,2,1)
        self.inception1a=InceptionModule(192,64,96,128,16,32,32)
        self.inception1b=InceptionModule(256,128,128,192,32,96,64)
        self.maxpool2=nn.MaxPool2d(3,2,1)

        self.inception2a=InceptionModule(480,192,96,208,16,48,64)
        self.inception2b=InceptionModule(512,160,112,224,24,64,64)
        self.inception2c=InceptionModule(512,128,128,256,24,64,64)
        self.inception2d=InceptionModule(512,112,144,288,32,64,64)
        self.inception2e=InceptionModule(528,256,160,320,32,128,128)
        self.maxpool3=nn.MaxPool2d(3,2,1)

        self.inception3a=InceptionModule(832,256,160,320,32,128,128)
        self.inception3b=InceptionModule(832,384,192,384,48,128,128)

        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, 1)

        self.inter1=InceptionInterModule(512,dropout_inter)
        self.inter2=InceptionInterModule(528,dropout_inter)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self,x:torch.Tensor):
        #Nx3x56x56
        x=self.conv1(x)
        #Nx3x56x56
        x=self.conv2(x)
        #Nx64x56x56
        x=self.conv3(x)
        #Nx192x56x56
        x=self.maxpool1(x)
        #Nx192x28x28
        x=self.inception1a(x)
        #Nx256x28x28
        x=self.inception1b(x)
        #Nx480x28x28
        x=self.maxpool2(x)
        #Nx480x14x14
        x=self.inception2a(x)
        #Nx512x14x14
        if self.training:
            aux1=self.inter1(x)#Nx1
        else:
            aux1=None
        x=self.inception2b(x)
        #Nx512x14x14
        x=self.inception2c(x)
        #Nx512x14x14
        x=self.inception2d(x)
        #Nx528x14x14
        if self.training:
            aux2=self.inter2(x)#Nx1
        else:
            aux2=None
        x=self.inception2e(x)
        #Nx832x14x14
        x=self.maxpool3(x)
        #Nx832x7x7
        x=self.inception3a(x)
        #Nx832x7x7
        x=self.inception3b(x)
        #Nx1024x7x7
        x=self.avgpool(x)
        #Nx1024x1x1
        x=nn.Flatten()(x)
        #Nx1024
        x=self.dropout(x)
        #Nx1024
        x=self.fc(x)
        #Nx1
        x=nn.Sigmoid()(x)
        #Nx1
        if self.training:
            return x,aux1,aux2
        else:
            return x



class ConvModule(nn.Module):
    """
    Convolutionnal module : Conv2d + BatchNorm2d + ReLu
    Input should be a (NxC1xH1xH1) tensor
    Output will be a (NxC2xH2xH2) tensor

    Args:
        in_c (int): number of channel of the input = C1
        out_c (int): number of channel of the output = C2
        kernel_size (int): kernel size for the convolution
        stride(int): stride for the convolution
        padding(int): padding for the convolution
    """
    def __init__(self, in_c: int, out_c: int, kernel_size: int, stride: int, padding= int):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding,  bias=False)
        self.batchnorm = nn.BatchNorm2d(out_c, eps=0.001)
        self.relu= nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x



class L2Pooling(nn.Module):
    """
    L2 Pooling Layer
    Input should be a (NxCxH1xH1) tensor
    Output will be a (NxCxH2xH2) tensor

    Args:
        kernel_size (int): kernel size used for the pooling
        stride (int): stride used for the pooling
        padding (int): padding used for the pooling
    
    """
    def __init__(self, kernel_size: int, stride: int, padding= int):
        super(L2Pooling, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,padding=padding)

    def forward(self, x: torch.Tensor):
        return torch.sqrt(self.pool(x ** 2))



class InceptionModule(nn.Module):
    """
    Inception Module as described in GoogLeNet paper
    Input should be a (NxC1xHxH) tensor
    Output will be a (NxC2xHxH) tensor, it is the result of concatenation
    of 4 branch, C2=ch1+ch3=ch5+chpool

    Args:
        in_c(int): number of channel of the input = C1
        ch1(int): number of channel of the output of the 1x1 branch
        ch3_red(int): number of channel of reduction conv in the 3x3 branch
        ch3(int): number of channel of the output of the 3x3 branch
        ch5_red(int): number of channel of reduction conv in the 5x5 branch
        ch5(int): number of channel of the output of the 5x5 branch
        chpool(int): number of channel of the output of the pooling branch
    """

    def __init__(self, in_c: int, ch1: int, ch3_red: int, ch3: int, ch5_red: int, ch5: int, chpool: int):

        super(InceptionModule, self).__init__()
        self.branch1=ConvModule(in_c,ch1,kernel_size=1,stride=1,padding=0)
        self.branch2=nn.Sequential(
            ConvModule(in_c,ch3_red,kernel_size=1,stride=1,padding=0),
            ConvModule(ch3_red,ch3,kernel_size=3,stride=1,padding=1)
        )
        self.branch3=nn.Sequential(
            ConvModule(in_c,ch5_red,kernel_size=1,stride=1,padding=0),
            ConvModule(ch5_red,ch5,kernel_size=5,stride=1,padding=2)
        )
        self.branch4=nn.Sequential(
            L2Pooling(kernel_size=3,stride=1, padding=1),
            ConvModule(in_c,chpool,kernel_size=1,stride=1,padding=0)
        )
    
    def forward(self,x: torch.Tensor):
        branch1= self.branch1(x)
        branch2= self.branch2(x)
        branch3= self.branch3(x)
        branch4= self.branch4(x)
        return torch.cat([branch1,branch2,branch3,branch4],1)

class InceptionInterModule(nn.Module):
    """
    Inception Intermediate Module, used only during training
    Input is a (NxCxHxH) tensor
    Output is a (Nx1) tensor
    Final layer is a sigmoid
    
    Args:
        in_c(int): number of channel of the input = C
        dropout(float):  Dropout probability for intermediate module
                prediction, should be 0<dropout<1
    """
    def __init__(self, in_c: int, dropout: float):
        super(InceptionInterModule, self).__init__()
        self.out=nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            ConvModule(in_c,128,kernel_size=1,stride=1,padding=0),
            nn.Flatten(),
            nn.Linear(2048,1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )

    def forward(self,x: torch.Tensor):
        x=self.out(x)
        return x
