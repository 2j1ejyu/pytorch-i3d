import torch
import torch.nn as nn

class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction_A(nn.Module):   # framewisely
    # B x 320 x T x 35 x 35 -> B x 1088 x T x 17 x 17
    def __init__(self, in_channels):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv3d(in_channels, 384, (1, 3, 3), stride=(1, 2, 2), padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv3d(256, 256, (1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            Conv3d(256, 384, (1, 3, 3), stride=(1, 2, 2), padding=0, bias=False)
        )
        self.branch_2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)                 # B x 384 x T x 17 x 17
        x1 = self.branch_1(x)                 # B x 384 x T x 17 x 17
        x2 = self.branch_2(x)                 # B x 320 x T x 17 x 17
        return torch.cat((x0, x1, x2), dim=1) # B x 1088 x T x 17 x 17
    
    
class Reduction_B(nn.Module):   # framewisely
    # B x 1088 x T x 17 x 17 -> B x 2080 x T x 8 x 8
    def __init__(self, in_channels):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv3d(256, 384, (1, 3, 3), stride=(1, 2, 2), padding=0, bias=False)
        )
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv3d(256, 288, (1, 3, 3), stride=(1, 2, 2), padding=0, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv3d(256, 288, (1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            Conv3d(288, 320, (1, 3, 3), stride=(1, 2, 2), padding=0, bias=False)
        )
        self.branch_3 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)                      # B x 384 x T x 8 x 8
        x1 = self.branch_1(x)                      # B x 288 x T x 8 x 8
        x2 = self.branch_2(x)                      # B x 320 x T x 8 x 8
        x3 = self.branch_3(x)                      # B x 1088 x T x 8 x 8
        return torch.cat((x0, x1, x2, x3), dim=1)  # B x 2080 x T x 8 x 8


class Stem(nn.Module):  
    def __init__(self, in_channels):
        # input: B x 3 x T x 299 x 299 (T=64)
        super(Stem, self).__init__()
        self.features = nn.Sequential(
            Conv3d(in_channels, 32, 3, stride=(1, 2, 2), padding=(1, 0, 0), bias=False), 
            # B x 32 x T x 149 x 149
            Conv3d(32, 32, 3, stride=1, padding=(1, 0, 0), bias=False),        
            # B x 32 x T x 147 x 147
            Conv3d(32, 64, 3, stride=1, padding=1, bias=False),          
            # B x 64 x T x 147 x 147
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=0),             #framewisely           
            # B x 64 x T x 73 x 73
            Conv3d(64, 80, 1, stride=1, padding=0, bias=False),          
            # B x 80 x T x 73 x 73
            Conv3d(80, 192, 3, stride=1, padding=(1, 0, 0), bias=False),         
            # B x 192 x T x 71 x 71
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=0),             #framewisely           
            # B x 192 x T x 35 x 35
        )
        self.branch_0 = Conv3d(192, 96, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(                                         #framewisely
            Conv3d(192, 48, 1, stride=1, padding=0, bias=False),
            Conv3d(48, 64, (1, 5, 5), stride=1, padding=(0, 2, 2), bias=False),
        )
        self.branch_2 = nn.Sequential(                                         #framewisely
            Conv3d(192, 64, 1, stride=1, padding=0, bias=False),
            Conv3d(64, 96, (1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            Conv3d(96, 96, (1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
        )
        self.branch_3 = nn.Sequential(                                         #framewisely
            nn.AvgPool3d((1, 3, 3), stride=1, padding=(0, 1, 1), count_include_pad=False),
            Conv3d(192, 64, 1, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        x = self.features(x)                        # B x 192 x T x 35 x 35
        x0 = self.branch_0(x)                       # B x 96 x T x 35 x 35
        x1 = self.branch_1(x)                       # B x 64 x T x 35 x 35
        x2 = self.branch_2(x)                       # B x 96 x T x 35 x 35
        x3 = self.branch_3(x)                       # B x 64 x T x 35 x 35
        return torch.cat((x0, x1, x2, x3), dim=1)   # B x 320 x T x 35 x 35


class Inception_ResNet_A(nn.Module):  
    # input: B x 320 x T x 35 x 35
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
        self.branch_0 = Conv3d(in_channels, 32, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv3d(32, 32, 3, stride=1, padding=1, bias=False)
        )
        self.branch_2 = nn.Sequential(
            Conv3d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv3d(32, 48, 3, stride=1, padding=1, bias=False),
            Conv3d(48, 64, 3, stride=1, padding=1, bias=False)
        )
        self.conv = nn.Conv3d(128, 320, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)                   # B x 32 x T x 35 x 35
        x1 = self.branch_1(x)                   # B x 32 x T x 35 x 35
        x2 = self.branch_2(x)                   # B x 64 x T x 35 x 35
        x_res = torch.cat((x0, x1, x2), dim=1)  # B x 128 x T x 35 x 35
        x_res = self.conv(x_res)                # B x 320 x T x 35 x 35
        return self.relu(x + self.scale * x_res)


class Inception_ResNet_B(nn.Module):    
    # input: B x 1088 x T x 17 x 17
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 128, 1, stride=1, padding=0, bias=False),
            Conv3d(128, 128, (7, 1, 1), stride=1, padding=(3, 0, 0), bias=False),
            Conv3d(128, 160, (1, 1, 7), stride=1, padding=(0, 0, 3), bias=False),
            Conv3d(160, 192, (1, 7, 1), stride=1, padding=(0, 3, 0), bias=False)
        )
        self.conv = nn.Conv3d(384, 1088, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)               # B x 192 x T x 35 x 35
        x1 = self.branch_1(x)               # B x 192 x T x 35 x 35
        x_res = torch.cat((x0, x1), dim=1)  # B x 384 x T x 35 x 35
        x_res = self.conv(x_res)            # B x 1088 x T x 35 x 35
        return self.relu(x + self.scale * x_res)


class Inception_ResNet_C(nn.Module):
    # input: B x 2080 x T x 8 x 8
    def __init__(self, in_channels, scale=1.0, activation=True):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv3d(192, 192, (3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),
            Conv3d(192, 224, (1, 1, 3), stride=1, padding=(0, 0, 1), bias=False),
            Conv3d(224, 256, (1, 3, 1), stride=1, padding=(0, 1, 0), bias=False)
        )
        self.conv = nn.Conv3d(448, 2080, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)                 # B x 192 x T x 35 x 35
        x1 = self.branch_1(x)                 # B x 256 x T x 35 x 35
        x_res = torch.cat((x0, x1), dim=1)    # B x 448 x T x 35 x 35
        x_res = self.conv(x_res)              # B x 2080 x T x 35 x 35
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res


class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=3, classes=8):
        super(Inception_ResNetv2, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(10):
            blocks.append(Inception_ResNet_A(320, 0.17))
        blocks.append(Reduction_A(320))
        for i in range(20):
            blocks.append(Inception_ResNet_B(1088, 0.10))
        blocks.append(Reduction_B(1088))
        for i in range(9):
            blocks.append(Inception_ResNet_C(2080, 0.20))
        blocks.append(Inception_ResNet_C(2080, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv3d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)                      # B x 2080 x T x 35 x 35
        x = self.conv(x)                          # B x 1536 x T x 35 x 35
        x = self.global_average_pooling(x)        # B x 1536 x T x 1 x 1
        x = x.squeeze(3).squeeze(3)               # B x 1536 x T
        x = self.linear(x.transpose(0,2,1))       # B x T x classes
        return x