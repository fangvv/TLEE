import os
import time
import shutil
import sys

import torch
import numpy as np
from loguru import logger
import torch.backends.cudnn as cudnn

from dataset.dataset import get_dataloader
from utils.utils import AverageMeter, accuracy
from model.tlee import get_policies
from model.tlee import BaseModel
from read_config import config

best_prec1 = 0
@logger.catch
def main():
    '''    
    Two stage to train the model, 
        1. the first stage is to train all the parameters except that of Gates, 
        2. the second stage is to train the parameters of Gates
        
    '''
    # 设置随机数种子，使得每次初始化权重都是固定的
    global best_prec1
    t1 = time.time()
    t1 = time.localtime(t1)
    t1 = time.strftime("%Y_%m_%d %H_%M_%S", t1)
    
    # model = BaseModel(config).cuda()
    # model.load_state_dict(torch.load('/home/wql/3d/FrameExit_run/log/train_stage_2_2021_10_23 23_08_25/ckpt.best.pth.tar')['state_dict'])
    # model = nn.DataParallel(model.cuda())

    # init setting
    cudnn.benchmark = True
    batch_size = 16
    print(f'batch_size:{batch_size}')

    train_loader = get_dataloader(config.data, batch_size=batch_size)
    val_loader = get_dataloader(config.data_val, batch_size=256)
    epochs = config.model.epochs
    best_prec1 = 0
    criterion = torch.nn.CrossEntropyLoss().cuda()
    stage_num = 'base'

    # 训练阶段
    print(f"-------------------------------BaseModel Train Stage-----------------------------------")
    model = BaseModel(config).cuda()
    lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(epochs):
        # if (epoch + 1) in [2, 4, 6]:
        #     lr *= 0.1
        #     for param_group in optimizer.param_groups:  #在每次更新参数前迭代更改学习率 
        #         param_group["lr"] = lr 
        prec1 = train_basemodel(train_loader, model, criterion, optimizer, epoch, lr)
        # model.load_state_dict(torch.load('./ckpt/base_hmdb51/ckpt.best.pth.tar')['state_dict'])
        if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch, lr)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            output_best = 'Best Prec@1:%.3f\n' % (best_prec1)
            print(output_best)
            check_rootfolders(t1, stage_num)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, True, stage_num, t1)

def train_basemodel(train_loader, model, criterion, optimizer, epoch, lr):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    for itr, batch in enumerate(train_loader):
        x, y = batch
        index = torch.tensor([5, 0, 9, 2, 7, 4, 6, 3, 8, 1], dtype=torch.long)
        # index = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=torch.long)
        x = torch.index_select(x, 1, index)
        x = x.to(device="cuda")
        z_previous = None
        y = y.cuda()
        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)
        loss_sum = 0
        for t in range(x.shape[1]):
            y_pred, z_previous = model(x[:, t].unsqueeze(dim=1), z_previous, t)
            loss = criterion(y_pred, y)
            loss_sum += loss
        loss_sum /= x.shape[1]
        prec1, prec5 = accuracy(y_pred, y, topk=(1,5))
        losses.update(loss_sum.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))
        top5.update(prec5.item(), x.size(0))
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        if itr != 0 and itr % 100 == 0:
            # flops, params = profile(model, x[:, t].unsqueeze(dim=1))
            # print(f"Flops:{flops / 1e9:.1f} GFlops, params:{params / 1e6:.1f} M")
            output = (
            f"Epoch: [{epoch:>3}][{itr:>4}/{len(train_loader):>4}] lr: {lr:.5f}\t"
            f"Loss: {losses.avg:.6f}  "
            f"Prec@1 {top1.avg:<3.3f}  "
            f"Prec@5 {top5.avg:<3.3f}"
            )
            print(output)

def validate(val_loader, model,criterion, epoch, lr):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    b_time = AverageMeter()

    model.eval()

    for itr, batch in enumerate(val_loader):
        x, y = batch
        y = y.cuda()
        index = torch.tensor([5, 0, 9, 2, 7, 4, 6, 3, 8, 1], dtype=torch.long)
        x = torch.index_select(x, 1, index)
        x = x.to(device="cuda")
        model.eval()

        z_previous = None
        loss_sum, prec1_sum, prec5_sum = 0, 0, 0
        with torch.no_grad():
            end = time.time()
            for t in range(x.shape[1]):
                y_pred, z_previous = model(x[:, t].unsqueeze(dim=1), z_previous, t)
                loss = criterion(y_pred, y)
                loss_sum += loss
                prec1, prec5 = accuracy(y_pred, y, topk=(1,5))
                prec1_sum += prec1
                prec5_sum += prec5
                
            batch_time = time.time() - end
            loss_sum /= x.shape[1]
            prec1_sum /= x.shape[1]
            prec5_sum /= x.shape[1]
            losses.update(loss_sum.item(), x.size(0))
            top1.update(prec1_sum.item(), x.size(0))
            top5.update(prec5_sum.item(), x.size(0))
            b_time.update(batch_time, x.size(0))    
            # if itr != 0 and itr % 100 == 0:
            output = (
                f"Test: [{itr:>4}/{len(val_loader):>4}]  "
                f"Loss: {losses.avg:.6f}  "
                f"Time: {b_time.avg:.5f}  "
                f"Prec@1 {top1.avg:<3.3f}  "
                f"Prec@5 {top5.avg:<3.3f}"
            )
            print(output)
    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    return top1.avg

def save_checkpoint(state, is_best, stage, t1):
    filename = f'/HOME/scz0831/run/3d/TLEE/ckpt/{stage}_{t1}/ckpt.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
def check_rootfolders(t1, stage):
    folder = f'/HOME/scz0831/run/3d/TLEE/ckpt/{stage}_{t1}'
    if not os.path.exists(folder):
        print('creating folder' + folder)
        os.mkdir(folder)
        print('creating success!')

if __name__ == "__main__":
    main()
    print("Done!")