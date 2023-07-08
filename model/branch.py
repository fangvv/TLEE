import torch.nn as nn

def conv3x3(inplanes, outplanes):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, bias=False)

def conv1x1(inplanes, outplanes):
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)

class Conv_Branch(nn.Module):
    def __init__(self, size, inplanes, outputdim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            conv1x1(inplanes=inplanes, outplanes=inplanes // 2),
            nn.BatchNorm2d(inplanes // 2),
            nn.ReLU(inplace=True),
            conv3x3(inplanes=inplanes // 2, outplanes=inplanes // 2),
            nn.BatchNorm2d(inplanes // 2),
            nn.ReLU(inplace=True),
            conv1x1(inplanes=inplanes // 2, outplanes=inplanes),
            nn.BatchNorm2d(inplanes),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes, outputdim)
    
    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = self.fc(x.view(x.shape[0], -1))
        return x
    
    
class FC_Branch(nn.Module):
    def __init__(self, size, inplanes, outputdim, fc_num=1) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.layers = nn.Sequential(
            nn.Linear(inplanes, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, outputdim)
        )
    
    def forward(self, x):
        x = self.pool(x)
        return self.layers(x.view(x.shape[0], -1))