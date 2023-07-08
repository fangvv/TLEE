import os
from random import randrange
import shutil

import torch
import pandas as pd
from read_config import config
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val      # 当前值
        self.sum += val * n # 正确数目
        self.count += n     # 总数
        self.avg = self.sum / self.count # 准确率

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk) # 1 or 5
    batch_size = target.size(0)  # 样本个数
    
    _, pred = output.topk(maxk, 1, True, True) # 返回每行前maxk最大的值
    pred = pred.t() # 转置
    # print(f"pred:{pred}m target:{target.view(1, -1).expand_as(pred)}")
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
def accuracy_gate(output, target, ):
    '''
    accuracy func for gate 
    '''
    batch_size = target.size(0)
    pred = torch.zeros_like(output)
    for i in range(pred.shape[0]):
        if output[i, 0] >= 0.5 :
            pred[i, 0] = 1
        elif output[i, 0] < 0.5:
            pred[i, 0] = 0
    pred = pred.t()
    # print(pred, output, target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in (1,):
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# tools
def save_checkpoint(state, is_best, stage, t1, super_folder=''):
    base_path = config.base_path
    feature_func = config.model.useFeatureFunc
    b = config.b
    model_name = config.model.name
    gate_func = ''
    if config.train_mode in [2, 'test']:
        gate_func = '_' + config.model.useGateFunc
    if super_folder:
        if stage != 2:
            filename = base_path + f'/{super_folder}/{stage}_{t1}_{feature_func}{gate_func}_{model_name}/ckpt.pth.tar'
        else:
            filename = base_path + f'/{super_folder}/{b}_{feature_func}{gate_func}/ckpt.pth.tar'
    else:
        if stage != 2:
            filename = base_path + f'/{stage}_{t1}_{feature_func}{gate_func}_{model_name}/ckpt.pth.tar'
        else:
            filename = base_path + f'/{b}_{feature_func}{gate_func}/ckpt.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

def check_rootfolders(t1, stage, super_folder=''):
    base_path = config.base_path
    feature_func = config.model.useFeatureFunc
    b = config.b
    model_name = config.model.name
    gate_func = ''
    if config.train_mode in [2, 'test']:
        gate_func = '_' + config.model.useGateFunc
    if super_folder:
        if stage != 2:
            folder = base_path + f'/{super_folder}/{stage}_{t1}_{feature_func}{gate_func}_{model_name}'
        else:
            folder = base_path + f'/{super_folder}/{b}_{feature_func}{gate_func}'

    else:
        if stage != 2:
            folder = base_path + f'/{stage}_{t1}_{feature_func}{gate_func}_{model_name}'
        else:
            folder = base_path + f'/{b}_{feature_func}{gate_func}'
    if not os.path.exists(folder):
        print('creating folder ' + folder)
        os.mkdir(folder)
        print('creating success!')
    return folder

def print_exp_data(
    is_train,
    epoch, 
    itr, 
    length,
    branch_num,
    num_frames,
    loss,
    loss_branch,
    top1_branch,
    top5_branch,
    top1,
    top5,
    lr,
    ):
    if is_train:
        output = (
            f"Epoch: [{epoch:>3}][{itr:>4}/{length:>4}] lr: {lr:.5f}\t"
            f"Loss: {loss.avg:.6f}  "
            f"Prec@1 {top1.avg:<3.3f}  "
            f"Prec@5 {top5.avg:<3.3f}"
        )
    else:
        output = (
            f"Test: [{itr:>4}/{length:>4}]  "
            f"Loss: {loss.avg:.6f}  "
            f"Prec@1 {top1.avg:<3.3f}  "
            f"Prec@5 {top5.avg:<3.3f}"
        )
    print(output)

    for branch_idx in range(branch_num):
        output = (
            f"Branch[{branch_idx + 1}]  "
            f"Loss: {loss_branch[branch_idx].avg:.5f}  "
            f"Prec@1 {top1_branch[branch_idx].avg:<3.3f}  "
            f"Prec@5 {top5_branch[branch_idx].avg:<3.3f}" 
        )
        print(output)

def compute_mac(data_dic, single_frame_mac):
    mac = 0
    for key in data_dic:
        frame, branch_idx = key
        # 每一个视频处理的帧数量 x 退出位置的MAC x 这样处理的视频的总数量
        # mac += frame * single_frame_mac[branch_idx - 1] * data_dic[key]
        # ((每一个视频处理的帧数量 - 1) x 最大MAC + 退出位置MAC) x 这样处理的视频的总数量
        mac += ((frame - 1) * single_frame_mac[-1] + single_frame_mac[branch_idx - 1]) * data_dic[key]
    return mac

def compute_mac_fine(data_dic, single_frame_mac):
    mac = 0
    video_nums = data_dic.shape[0]
    frames = data_dic.shape[1] - 2
    for i in range(video_nums):
        for j in range(frames):
            exit_loc = data_dic[i, j]
            if exit_loc > 0:
                mac += single_frame_mac[exit_loc - 1]
            else:
                break
    return mac

def save_info(path, exit_info, classes_info):
    df_exit_info = pd.DataFrame(
        ([
            [key[0], key[1], exit_info[key]]
            for key in exit_info
        ])
    )

    with pd.ExcelWriter(path, mode='a') as writer:
        df_exit_info.to_excel(writer, sheet_name='exit_info', index=False, header=False)
        df_index = pd.DataFrame(
            [
                [key[0], key[1]]
                for key in exit_info
            ]
        )
        df_index.to_excel(writer, sheet_name='classes_info', index=False, header=False, startcol=0)
        for idx, item in enumerate(classes_info):
            df_classes_info = pd.DataFrame(
                [
                    [item[key]]
                    for key in item
                ],
            )
            df_classes_info.to_excel(
                writer,
                sheet_name='classes_info',
                index=False,
                header=False,
                startcol=idx+2,
            )
    print(f'{path} saved!')


def save_info_fine(path, exit_info):
    df_exit_info = pd.DataFrame(
        ([
            [exit_info[i, j] for j in range(exit_info.shape[1])]
            for i in range(exit_info.shape[0])
        ])
    )

    with pd.ExcelWriter(path, mode='a') as writer:
        df_exit_info.to_excel(writer, sheet_name='exit_info', index=False, header=False)
        # save classes info
        # df_classes_info = pd.DataFrame(
        #     ([

        #     ])
        # )
    print(f'{path} saved!')