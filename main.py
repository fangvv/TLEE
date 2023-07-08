# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import time
import math

import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
from loguru import logger
import torch.backends.cudnn as cudnn
from model.tlee import get_policies

from dataset.dataset import get_dataloader
from model.tlee import TLEEModel
from utils.utils import (
    AverageMeter, accuracy, accuracy_gate, save_checkpoint,
    check_rootfolders, print_exp_data, compute_mac, save_info, 
    compute_mac_fine, save_info_fine,
    )
from read_config import config
from tools.lr_schedule import lr_schedule

# 设置随机数种子，使得每次初始化权重都是固定的
if config.seed:
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

best_prec1 = 0
model_name = config.model.name
num_frames = config.data.num_frames # 10
branch_num = config.model.branch_num
num_classes = config.model.num_classes
input_order = config.model.input_order10
prec1Thresh = 40

print(
    f"epochs: {config.model.epochs}\n"
    f"backbone: {model_name}\n"
    f"num_frames: {config.data.num_frames}\n"
    f"crm_num: {config.model.crm_num}\n"
    f"branchScale: {config.model.branch_scale}\n"
    f"FeatureFunc: {config.model.useFeatureFunc}\n"
    f"useGateFunc: {config.model.useGateFunc}\n"
    f"branch_num: {branch_num}\n"
    )

cudnn.benchmark = True
# init setting
if config.train_mode == 2:
    batch_size = 256
    val_batch_size = 1
elif config.train_mode == 'test':
    batch_size = 1
    val_batch_size = 1
elif config.train_mode == 1 or config.train_mode == 1.1:
    batch_size = 8
    if config.data.name == 'hmdb51':
        batch_size = 8
    if config.model.branch_num == 0:
        val_batch_size = 1
    else:
        val_batch_size = 256

print(f'batch_size: {batch_size} val_batch_size: {val_batch_size}')
train_loader = get_dataloader(config.data, batch_size=batch_size)
val_loader = get_dataloader(config.data_val, batch_size=val_batch_size)
epochs = config.model.epochs
best_prec1 = 0
criterion1 = torch.nn.CrossEntropyLoss().cuda()

t1 = time.time()
t1 = time.localtime(t1)
t1 = time.strftime("%Y_%m_%d %H_%M_%S", t1)

@logger.catch
def main():
    '''    
    Two stage to train the model, 
        1. the first stage is to train all the parameters except that of Gates, 
        2. the second stage is to train the parameters of Gates
        
    '''
    global best_prec1
    logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s')

    logging.info(f'Time: {t1}')
    # 训练阶段
    print(f'Train Mode: {config.train_mode}')
    if config.train_mode in [2, 'test']:
        print(f"b: {config.b}")
    
    if config.train_mode == 1.1:
        model = TLEEModel(config, is_train=True, branch_num=branch_num).cuda()
        train_w_freeze_branch(model)
        if config.model.branch_num != 0:
            train_w_freeze_backbone(model)
        # train_1(model)

    elif config.train_mode == 1:
        model = TLEEModel(config, is_train=True, branch_num=branch_num).cuda()
        train_1(model)

    elif config.train_mode == 2:
        train_2()

    elif config.train_mode == 'test':
        inference_mode()
    else:
        raise NotImplementedError(f"Please Check the train_mode {config.train_mode}!")

# train settings
def train_1(model):
    '''
    joint-training
    '''
    print(f"-------------------------------First training Stage-----------------------------------")
    lr = 1e-3
    stage_num = 1
    global best_prec1
    if config.show_model:
        get_policies(model, show_model=config.show_model)
        return
    params = get_policies(model, stage_num)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(epochs):
        if config.model.useLRSchedule:
            lr_schedule(config, lr, optimizer, epoch)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
        train_base(train_loader, model, criterion1, optimizer, epoch, lr)

        torch.save(model.state_dict(), './test.pth.tar')
        print('SAVE TEST MODEL SUCCESS')
        if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
            print("#" * 31 + "Validate begin ..." + "#" * 31)
            # model.load_state_dict(torch.load('ckpt/1_2022_01_16 20_07_56/ckpt.best.pth.tar')['state_dict'])
            prec1 = validate(val_loader, model, criterion1, epoch, lr)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            output_best = 'Best Prec@1:%.3f\n' % (best_prec1)
            print(output_best)
            
            if best_prec1 > prec1Thresh:
                file_path = check_rootfolders(t1, stage_num)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, stage_num, t1)
                with open(f'{file_path}/info.log', 'w') as f:
                    f.write(
                        f'epoch: {epoch+1}\n'
                        f'best_prec1: {best_prec1}\n'
                        f'stage num: {stage_num}'
                        )

def train_2():
    '''
    门控训练
    '''
    print(f"-------------------------------Second training Stage-----------------------------------")
    global best_prec1
    model = TLEEModel(config, is_train=True, branch_num=branch_num).cuda()
    ckpt_path = config.test_path
    ckpt = torch.load('ckpt/' + ckpt_path + '/ckpt.best.pth.tar')

    best_prec1 = ckpt['best_prec1']
    print(f'The ckpt file is : {ckpt_path}')
    print(f"This ckpt file's best_prec1: {best_prec1:.4f}")
    
    model.load_state_dict(ckpt['state_dict'], strict=False)

    lr = 1e-3
    stage_num = 2
    criterion2 = torch.nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.SGD(model.exit_selector.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    s_time = time.time()
    for epoch in range(3):
        if config.model.useLRSchedule:
            lr_schedule(config, lr, optimizer, epoch)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
        train_gate(train_loader, model, criterion1, criterion2, optimizer, epoch, lr, eval(config.b))
        if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
            # prec1 = validate(val_loader, model, criterion1, epoch, lr)
            file_path = check_rootfolders(t1, stage_num, super_folder=ckpt_path)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
                'b': config.b,
            }, True, stage_num, t1, super_folder=ckpt_path)
    print(f"Costs {(time.time() - s_time):.1f}s")

def inference_mode():
    '''
    推理模式
    '''
    print(f"-------------------------------Test Stage-----------------------------------")
    model = TLEEModel(config, is_train=False, branch_num=branch_num).cuda()
    ckpt_path = config.test_path
    ckpt = torch.load('ckpt/' + ckpt_path + '/ckpt.best.pth.tar')
    best_prec1 = ckpt['best_prec1']
    if 'b' in ckpt:
        b = ckpt['b']
    else:
        b = 'Not set, may be you use the train_stage 1 model'
    print(f'Test ckpt path is : {ckpt_path}')
    print(f"This ckpt file's best_prec1: {best_prec1}")
    print(f"b: {b}")
    print(f"exit_thresh: {config.model.exit_thresh}")

    model.load_state_dict(ckpt['state_dict'], strict=False)
    # prec1, results, mac, exit_info, classes_info = test(val_loader, model, criterion1, b=b)
    prec1, results, mac, exit_info, classes_info = test_fine(val_loader, model, criterion1, b=b)
    # save info to excel
    with pd.ExcelWriter(f'ckpt/{ckpt_path}/info.xlsx', ) as writer:
        df_base_info = pd.DataFrame(
            [[ckpt_path, b, best_prec1, prec1, mac]],
            columns=['Path', 'b', 'base_best_prec1', 'final_prec1', 'mac']
        )
        df_base_info.to_excel(
            writer, 
            sheet_name='base_info', 
            index=False,
            )
    # save_info(f'ckpt/{ckpt_path}/info.xlsx', exit_info, classes_info)
    save_info_fine(f'ckpt/{ckpt_path}/info.xlsx', exit_info)

    with open('ckpt/' + ckpt_path + '/results.log', 'w+') as f:
        f.write(results)
        print("Results write success!")

def train_w_freeze_branch(model):
    '''
    backbone and last classifer training
    '''
    print("-"*31 + "Training backbone and last classifier" + "-"*31)
    lr = 1e-3
    stage_num = 1.1
    global best_prec1
    

    params = get_policies(model, stage_num, show_model=False)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(epochs):
        if config.model.useLRSchedule:
            lr_schedule(config, lr, optimizer, epoch)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
        train_policy_1(train_loader, model, criterion1, optimizer, epoch, lr)
        torch.save(model.state_dict(), './test.pth.tar')
        print('SAVE TEST MODEL SUCCESS')
        
        if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
            print("#" * 31 + "Test begin ..." + "#" * 31)
            prec1 = validate(val_loader, model, criterion1, epoch, lr)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            output_best = 'Best Prec@1:%.3f\n' % (best_prec1)
            print(output_best)
        
            if best_prec1 > prec1Thresh:
                file_path = check_rootfolders(t1, stage_num)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, stage_num, t1)
                with open(f'{file_path}/info.log', 'w') as f:
                    f.write(
                        f'epoch: {epoch+1}\n'
                        f'best_prec1: {best_prec1}\n'
                        f'stage num: {stage_num}'
                        )

def train_w_freeze_backbone(model):
    '''
    train branch
    '''
    global best_prec1
    print("-"*31 + "Training Branches" + "-"*31)
    lr = 1e-3
    start_epoch = 0
    stage_num = 1.2
    best_prec1 = 0
    params = get_policies(model, stage_num, show_model=False)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    # resume
    if config.resume == True:
        resume_path = config.resume_path
        assert isinstance(resume_path, str)
        ckpt = torch.load(resume_path + '/ckpt.best.pth.tar')
        # TODO 保存当前训练阶段的epoch
        # start_epoch = ckpt['epoch']

        print(f"finetune from {resume_path}...")
        model.load_state_dict(ckpt['state_dict'])
        
    for epoch in range(start_epoch, epochs * 2):
        if config.model.useLRSchedule:
            lr_schedule(config, lr, optimizer, epoch)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
        train_policy_2(train_loader, model, criterion1, optimizer, epoch, lr)
        torch.save(model.state_dict(), './test.pth.tar')
        print('SAVE TEST MODEL SUCCESS')
        
        if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
            print("#" * 31 + "Test begin ..." + "#" * 31)
            prec1 = validate(val_loader, model, criterion1, epoch, lr)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            output_best = 'Best Prec@1:%.3f\n' % (best_prec1)
            print(output_best)
        
            if best_prec1 > prec1Thresh:
                file_path = check_rootfolders(t1, stage_num,)
                is_best = True # to save the branch model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, stage_num, t1)
                with open(f'{file_path}/info.log', 'w') as f:
                    f.write(
                        f'epoch: {epoch+1}\n'
                        f'best_prec1: {best_prec1}\n'
                        f'stage num: {stage_num}'
                        )

def train_base(train_loader, model, criterion, optimizer, epoch, lr):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # 每一个分支的loss, prec1, prec5
    loss_branch = [AverageMeter() for _ in range(branch_num + 1)]
    top1_branch = [AverageMeter() for _ in range(branch_num + 1)]
    top5_branch = [AverageMeter() for _ in range(branch_num + 1)]

    model.train()
    end = time.time()
    for itr, batch in enumerate(train_loader):
        x, y = batch
        index = torch.tensor([5, 0, 9, 2, 7, 4, 6, 3, 8, 1], dtype=torch.long)
        # index = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=torch.long)
        x = torch.index_select(x, 1, index)
        x = x.cuda()
        y = y.cuda()
        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)

        loss_frame_all = []
        prec1_frame_all = []
        prec5_frame_all = []

        z_previous = None
        model.clearPreviousFeature()
        for t in range(x.shape[1]):
            # 每个帧的loss, prec1, prec5
            mean_loss_per_frame, mean_prec1_per_frame, mean_prec5_per_frame = 0, 0, 0
            for branch_idx in range(branch_num + 1): # branch_num个分支+一个完整网络
                y_pred, z_previous = model(x[:, t].unsqueeze(dim=1), z_previous, t, branch_idx)
                loss = criterion(y_pred, y)
                prec1, prec5 = accuracy(y_pred, y, topk=(1,5))
                loss_branch[branch_idx].update(loss.item(), x.size(0))
                top1_branch[branch_idx].update(prec1.item(), x.size(0))
                top5_branch[branch_idx].update(prec5.item(), x.size(0))
                mean_loss_per_frame += loss
                mean_prec1_per_frame += prec1
                mean_prec5_per_frame += prec5

            mean_loss_per_frame /= branch_num + 1
            mean_prec1_per_frame /= branch_num + 1
            mean_prec5_per_frame /= branch_num + 1

            loss_frame_all.append(mean_loss_per_frame)
            prec1_frame_all.append(mean_prec1_per_frame)
            prec5_frame_all.append(mean_prec5_per_frame)

        loss_all = sum(loss_frame_all) / x.shape[1]
        prec1_all = sum(prec1_frame_all) / x.shape[1]
        prec5_all = sum(prec5_frame_all) / x.shape[1]
        
        losses.update(loss_all.item(), x.size(0))
        top1.update(prec1_all.item(), x.size(0))
        top5.update(prec5_all.item(), x.size(0))

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        
        if itr != 0 and itr % 100 == 0:
            exp_data = {
                'is_train': True,
                'epoch': epoch,
                'itr': itr,
                'length': len(train_loader),
                'branch_num': branch_num + 1,
                'num_frames': num_frames,
                'loss': losses,
                'loss_branch': loss_branch,
                'top1_branch': top1_branch,
                'top5_branch': top5_branch,
                'top1': top1,
                'top5': top5,
                'lr': lr,
                }
            print_exp_data(**exp_data)

def train_gate(train_loader, model, criterion1, criterion2, optimizer, epoch, lr, b):
    '''
    : criterion1: nn.CrossEntropyLoss()
    : criterion2: nn.BCEWithLogitsLoss()
    : b: hyperparameter to control the accuracy-latency trade-off
    '''
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    end = time.time()
    for itr, batch in enumerate(train_loader):
        x, y = batch
        index = torch.tensor(input_order, dtype=torch.long)
        x = torch.index_select(x, 1, index)
        x = x.to(device="cuda")
        z_previous = None
        prev_frame_f = None
        y = y.cuda()
        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)
        loss_all = 0
        prec1_all = 0
        for t in range(x.shape[1]):
            # 获取第t个门控网络
            gate = getattr(model, 'exit_selector')[0]
            gate.train()
            for branch_idx in range(branch_num + 1):
                threshold = b * math.exp(t / (2 + branch_idx + 1))
                with torch.no_grad():    
                    y_pred, z_previous_cur = model(x[:, t].unsqueeze(dim=1), z_previous, t, branch_idx=branch_idx)
                    loss1 = criterion1(y_pred, y)
                if loss1.item() > threshold:
                    y_g = torch.zeros((x.shape[0], 1)).float().cuda()
                else:
                    y_g = torch.ones((x.shape[0], 1)).float().cuda()

                # TODO 修改为前一帧和当前帧的特征cat
                if t > 0:
                    input_z = [z_previous_cur.view(z_previous_cur.shape[0], -1, 1, 1), prev_frame_f.view(prev_frame_f.shape[0], -1, 1, 1)]
                else:
                    input_z = [z_previous_cur.view(z_previous_cur.shape[0], -1, 1, 1), z_previous_cur.view(z_previous_cur.shape[0], -1, 1, 1)]
                out_g = gate(input_z)
                # if t == 1:
                #     print(t, nn.Sigmoid()(out_g)[0])
                # else:
                #     y_t_probs = torch.nn.Softmax(dim=1)(z_previous_cur)
                #     exit_door = torch.max(y_t_probs, dim=1)[0] > 0.99
                #     out_g = torch.where(exit_door == False, torch.tensor(0).float().cuda(), torch.tensor(1).float().cuda()).unsqueeze(1).cuda()
                z_previous = z_previous_cur
                loss = criterion2(out_g, y_g)
                prec1 = accuracy_gate(nn.Sigmoid()(out_g), y_g,)[0]
            prev_frame_f = z_previous_cur
            loss_all += loss
            prec1_all += prec1

        loss_all /= x.shape[1]
        prec1_all /= x.shape[1]
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        losses.update(loss_all.item(), x.size(0))
        top1.update(prec1_all.item(), x.size(0))
        
        if itr % 100 == 0 or itr == len(train_loader) - 1:
            # print(f'exit_selector.weight: {model.exit_selector[0].conv1.weight}')
            output = (
                        f'Epoch: [{epoch}][{itr}/{len(train_loader)}], lr: {lr:.5f}\t'
                        f'Loss {losses.avg:.4f}\t'
                        f'Prec@1 {top1.avg:.3f}\t'
                      )
            print(output)

def train_policy_1(train_loader, model, criterion, optimizer, epoch, lr):
    '''
    Train the backbone
    '''
    losses = AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()

    model.train()
    for itr, batch in enumerate(train_loader):
        x, y = batch
        index = torch.tensor(input_order, dtype=torch.long)
        # index = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=torch.long)
        x = torch.index_select(x, 1, index)
        x = x.cuda()
        z_previous = None
        y = y.cuda()
        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)

        loss_all, prec1_all, prec5_all = 0, 0, 0
        for t in range(x.shape[1]):
            y_pred, z_previous = model(x[:, t].unsqueeze(dim=1), z_previous, t, branch_idx=branch_num)
            loss = criterion(y_pred, y)
            prec1, prec5 = accuracy(y_pred, y, topk=(1,5))
            loss_all += loss
            prec1_all += prec1
            prec5_all += prec5

        loss_all /= x.shape[1]
        prec1_all /= x.shape[1]
        prec5_all /= x.shape[1]
        losses.update(loss_all.item(), x.size(0))
        top1.update(prec1_all.item(), x.size(0))
        top5.update(prec5_all.item(), x.size(0))

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        if itr != 0 and itr % 100 == 0:
            output = (
                f"Train_1:[{epoch:>3}][{itr:>4}/{len(train_loader):>4}] lr: {lr:.5f} "
                f"Loss: {losses.avg:.6f}\t"
                f"Prec@1 {top1.avg:<3.3f} Prec@5 {top5.avg:<3.3f}"
                )
            print(output)

def train_policy_2(train_loader, model, criterion, optimizer, epoch, lr):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # 每一个分支的loss, prec1, prec5
    loss_branch = [AverageMeter() for _ in range(branch_num + 1)]
    top1_branch = [AverageMeter() for _ in range(branch_num + 1)]
    top5_branch = [AverageMeter() for _ in range(branch_num + 1)]

    model.eval() # 固定主干部分的bn层
    end = time.time()
    for itr, batch in enumerate(train_loader):
        x, y = batch
        index = torch.tensor(input_order, dtype=torch.long)
        # index = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=torch.long)
        x = torch.index_select(x, 1, index)
        x = x.cuda()
        y = y.cuda()
        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)

        loss_frame_all = []
        prec1_frame_all = []
        prec5_frame_all = []

        z_previous = None
        for t in range(x.shape[1]):
            # 每个帧的loss, prec1, prec5
            mean_loss_per_frame, mean_prec1_per_frame, mean_prec5_per_frame = 0, 0, 0
            for branch_idx in range(branch_num + 1): # branch_num个分支+一个完整网络
                y_pred, z_previous = model(x[:, t].unsqueeze(dim=1), z_previous, t, branch_idx)
                loss = criterion(y_pred, y)
                prec1, prec5 = accuracy(y_pred, y, topk=(1,5))
                loss_branch[branch_idx].update(loss.item(), x.size(0))
                top1_branch[branch_idx].update(prec1.item(), x.size(0))
                top5_branch[branch_idx].update(prec5.item(), x.size(0))
                if branch_idx != branch_num:
                    mean_loss_per_frame += loss
                    mean_prec1_per_frame += prec1
                    mean_prec5_per_frame += prec5

            mean_loss_per_frame /= branch_num
            mean_prec1_per_frame /= branch_num
            mean_prec5_per_frame /= branch_num

            loss_frame_all.append(mean_loss_per_frame)
            prec1_frame_all.append(mean_prec1_per_frame)
            prec5_frame_all.append(mean_prec5_per_frame)

        loss_all = sum(loss_frame_all) / x.shape[1]
        prec1_all = sum(prec1_frame_all) / x.shape[1]
        prec5_all = sum(prec5_frame_all) / x.shape[1]
        
        losses.update(loss_all.item(), x.size(0))
        top1.update(prec1_all.item(), x.size(0))
        top5.update(prec5_all.item(), x.size(0))

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        
        if itr != 0 and itr % 100 == 0:
            exp_data = {
                'is_train': True,
                'epoch': epoch,
                'itr': itr,
                'length': len(train_loader),
                'branch_num': branch_num + 1,
                'num_frames': num_frames,
                'loss': losses,
                'loss_branch': loss_branch,
                'top1_branch': top1_branch,
                'top5_branch': top5_branch,
                'top1': top1,
                'top5': top5,
                'lr': lr,
                }
            print_exp_data(**exp_data)
   
def validate(val_loader, model, criterion, epoch, lr):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # 每个分支的信息
    loss_branch = [AverageMeter() for _ in range(branch_num + 1)]
    top1_branch = [AverageMeter() for _ in range(branch_num + 1)]
    top5_branch = [AverageMeter() for _ in range(branch_num + 1)]
    for itr, batch in enumerate(val_loader):
        x, y = batch
        y = y.cuda()
        index = torch.tensor(input_order, dtype=torch.long)
        x = torch.index_select(x, 1, index)
        x = x.to(device="cuda")
        # 所有分支的分类信息
        loss_frame_all = []
        prec1_frame_all = []
        prec5_frame_all = []
        model.eval()
        z_previous = None
        with torch.no_grad():
            for t in range(x.shape[1]):
                mean_loss_per_frame, mean_prec1_per_frame, mean_prec5_per_frame = 0, 0, 0
                for branch_idx in range(branch_num + 1):
                    y_pred, z_previous = model(x[:, t].unsqueeze(dim=1), z_previous, t, branch_idx)
                    loss = criterion(y_pred, y)
                    prec1, prec5 = accuracy(y_pred, y, topk=(1,5))
                    loss_branch[branch_idx].update(loss.item(), x.size(0))
                    top1_branch[branch_idx].update(prec1.item(), x.size(0))
                    top5_branch[branch_idx].update(prec5.item(), x.size(0))
                    mean_loss_per_frame += loss
                    mean_prec1_per_frame += prec1
                    mean_prec5_per_frame += prec5
                
                # 每个分支的平均信息
                mean_loss_per_frame /= branch_num + 1
                mean_prec1_per_frame /= branch_num + 1
                mean_prec5_per_frame /= branch_num + 1

                loss_frame_all.append(mean_loss_per_frame)
                prec1_frame_all.append(mean_prec1_per_frame)
                prec5_frame_all.append(mean_prec5_per_frame)
            # 所有分支（包括最后的分类器）的平均精度
            loss_all = sum(loss_frame_all) / x.shape[1]
            prec1_all = sum(prec1_frame_all) / x.shape[1]
            prec5_all = sum(prec5_frame_all) / x.shape[1]
            
            losses.update(loss_all.item(), x.size(0))
            top1.update(prec1_all.item(), x.size(0))
            top5.update(prec5_all.item(), x.size(0))
                
        if (itr != 0 and itr % 100 == 0) or itr == len(val_loader) - 1:
            exp_data = {
            'is_train': False,
            'epoch': epoch,
            'itr': itr,
            'length': len(val_loader),
            'branch_num': branch_num + 1,
            'num_frames': num_frames,
            'loss': losses,
            'loss_branch': loss_branch,
            'top1_branch': top1_branch,
            'top5_branch': top5_branch,
            'top1': top1,
            'top5': top5,
            'lr': lr,
            }
            print_exp_data(**exp_data)
    output = (
        f'Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.5f} '
        )
    print(output)
    return top1_branch[-1].avg

def test(val_loader, model, criterion, b):
    '''
    without auxilary exit thresh
    '''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    classes_info = [{} for _ in range(num_classes)]

    model.eval()
    exit_stats_count = 0
    exit_stats_dic = {}

    for item in range(num_classes):
        for frame in range(num_frames):
            for branch_idx in range(branch_num + 1):
                if item == 0:
                    exit_stats_dic[(frame + 1, branch_idx + 1)] = 0
                classes_info[item][(frame + 1, branch_idx + 1)] = 0
   

    end = time.time()
    single_frame_mac = config.model.single_frame_mac                  # if 'resnet50' in config.model.backbone.name else 0
    for itr, batch in enumerate(val_loader):
        x, y = batch
        y = y.cuda()
        index = torch.tensor(input_order, dtype=torch.long)
        x = torch.index_select(x, 1, index)
        x = x.to(device="cuda")

        z_previous = None
        with torch.no_grad():
            for t in range(x.shape[1]):
                if config.model.type == 'ts': # 时空退出
                    for branch_idx in range(branch_num + 1):
                        y_pred, z_previous, _ = model(x[:, t].unsqueeze(dim=1), z_previous, t, branch_idx)
                        if y_pred is not None:  # exit if true
                            classes_info[y.item()][(t + 1, branch_idx + 1)] += 1
                            exit_stats_dic[(t + 1, branch_idx + 1)] += 1
                            loss = criterion(y_pred, y)
                            exit_stats_count += 1
                            prec1, prec5 = accuracy(y_pred, y, topk=(1,5))
                            losses.update(loss.item(), x.size(0))
                            top1.update(prec1.item(), x.size(0))
                            top5.update(prec5.item(), x.size(0))
                            break
                    else:
                        continue
                    break
                elif config.model.type == 't': # 时间退出
                    y_pred, z_previous = model(x[:, t].unsqueeze(dim=1), z_previous, t, branch_num)
                    if y_pred is not None:  # exit if true
                        classes_info[y.item()][(t + 1, branch_num + 1)] += 1
                        exit_stats_dic[(t + 1, branch_num + 1)] += 1
                        loss = criterion(y_pred, y)
                        exit_stats_count += 1
                        prec1, prec5 = accuracy(y_pred, y, topk=(1,5))
                        losses.update(loss.item(), x.size(0))
                        top1.update(prec1.item(), x.size(0))
                        top5.update(prec5.item(), x.size(0))
                        break
            batch_time.update(time.time() - end)
            end = time.time()
            if itr % 100 == 0 or itr == len(val_loader) - 1:
                print(
                f"""Test: [{itr}/{len(val_loader)}]
                Time: {batch_time.avg:.4f}
                Loss: {losses.avg:.4f}
                Prec@1 {top1.avg:.3f}
                Prec@5 {top5.avg:.3f}
                exit_stats: {exit_stats_dic} ({exit_stats_count})
                """
                )
    
    mac = compute_mac(exit_stats_dic, single_frame_mac[-(branch_num + 1):]) / exit_stats_count

    output = f"""Testing Results: Prec@1 {top1.avg:.3f} 
        b {b}
        Prec@5 {top5.avg:.3f} 
        Time: {batch_time.avg}
        Loss {losses.avg:.5f} 
        GFlops: {mac}
        exit_states: {exit_stats_dic}
        """
    print(output)
    return top1.avg, output, mac, exit_stats_dic, classes_info

def test_fine(val_loader, model, criterion, b):
    '''
    with auxilary exit thresh
    '''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    classes_info = {}
    for i in range(num_classes):
        classes_info[i] = []

    model.eval()

    exit_stats_count = 0
    exit_stats_dic = {}
    exit_data = np.zeros((len(val_loader), num_frames + 2), dtype=int) # + 2 means true classInd and pred classInd


    single_frame_mac = config.model.single_frame_mac                  # if 'resnet50' in config.model.backbone.name else 0
    for itr, batch in enumerate(val_loader):
        x, y = batch
        y = y.cuda()
        index = torch.tensor(input_order, dtype=torch.long)
        x = torch.index_select(x, 1, index)
        x = x.to(device="cuda")

        z_previous = None
        with torch.no_grad():
            end = time.time()
            for t in range(x.shape[1]):
                if config.model.type == 'ts': # 时空退出
                    for branch_idx in range(branch_num + 1):
                        y_pred, z_previous, branch_exit = model(x[:, t].unsqueeze(dim=1), z_previous, t, branch_idx)
                        if y_pred is not None:  # exit if true
                            exit_data[itr, t] = branch_idx + 1
                            exit_data[itr, -2] = y.item()
                            exit_data[itr, -1] = y_pred.topk(1, 1, True, True)[1]
                            loss = criterion(y_pred, y)
                            exit_stats_count += 1
                            prec1, prec5 = accuracy(y_pred, y, topk=(1,5))
                            losses.update(loss.item(), x.size(0))
                            top1.update(prec1.item(), x.size(0))
                            top5.update(prec5.item(), x.size(0))
                            break
                        if branch_exit:
                            exit_data[itr, t] = branch_idx + 1
                            break
                    else:
                        exit_data[itr, t] = branch_num + 1
                        continue
                    if branch_exit and y_pred is None:
                        continue
                    else:
                        break
                elif config.model.type == 't': # 时间退出
                    y_pred, z_previous, _ = model(x[:, t].unsqueeze(dim=1), z_previous, t, branch_num)
                    exit_data[itr, t] = 3
                    if y_pred is not None:  # exit if true
                        exit_data[itr, -2] = y.item()
                        exit_data[itr, -1] = y_pred.topk(1, 1, True, True)[1]
                        loss = criterion(y_pred, y)
                        exit_stats_count += 1
                        prec1, prec5 = accuracy(y_pred, y, topk=(1,5))
                        losses.update(loss.item(), x.size(0))
                        top1.update(prec1.item(), x.size(0))
                        top5.update(prec5.item(), x.size(0))
                        break
            batch_time.update(time.time() - end)
            if itr % 100 == 0 or itr == len(val_loader) - 1:
                print(
                f"""Test: [{itr}/{len(val_loader)}]
                Time: {batch_time.avg:.4f}
                Loss: {losses.avg:.4f}
                Prec@1 {top1.avg:.3f}
                Prec@5 {top5.avg:.3f}
                exit_stats: {exit_data} ({exit_stats_count})
                """
                )

    with open('./exit_info.txt', 'w+') as f:
        f.write(str(exit_data.tolist()))

    mac = compute_mac_fine(exit_data, single_frame_mac[-(branch_num + 1):]) / exit_stats_count

    output = f"""Testing Results: Prec@1 {top1.avg:.3f} 
        b {b}
        Prec@5 {top5.avg:.3f} 
        Time: {batch_time.avg}
        Loss {losses.avg:.5f} 
        GFlops: {mac}
        exit_states: {exit_data}
        """
    print(output)
    return top1.avg, output, mac, exit_data, classes_info

if __name__ == "__main__":
    main()
    print("Done!")