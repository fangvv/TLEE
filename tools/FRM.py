import torch
import torch.nn as nn

class AttentionFRM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
        self.testfrm = nn.Sequential(
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x0, x1):
        xx0 = self.testfrm(x0)
        xx1 = self.testfrm(x1)
        return (xx0 + xx1)

class EMAFRM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.beta = nn.Parameter(torch.tensor(0.999)).cuda()
    
    def forward(self, x1, x2):
        return self.beta * x2 + (1 - self.beta) * x1


class AveragePooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, y):
        x = torch.cat((x.unsqueeze(1), y.unsqueeze(1)), dim=1)
        return x.mean(dim=1)

class NaiveAdd(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, y):
        return torch.abs(x) + torch.abs(y)
