import torchvision.models as models
import torch

import time

device = torch.device('cpu')
model = models.mobilenet_v2(False).to(device)
x = torch.randn(1, 3, 224, 224).to(device)
s = 0
for i in range(110):
    end = time.time()
    for t in range(1):
        _ = model(x)
    start = time.time()
    if i > 9:
        s += start - end
    print(f"Time: {start - end:.4f}s")
print(f"average time: {s / 100:.4f}")
