# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
import sys
sys.path.append(".")
import random, time

import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from tools.FRM import AttentionFRM, EMAFRM, AveragePooling, NaiveAdd
from tools.random_exit import random_exit
from read_config import config as testConfig

# 最大池化
class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        # print(f"z_t: {x.shape} | z_previous: {y.shape}")
        x = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1)), dim=1)
        # print(f"x: {x.shape}")
        x = x.max(dim=1)[0]
        return x

class FeatureReuseModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x, y):
        x = x.unsqueeze(dim=1)
        for feature in y:
            x = torch.cat((x, feature.unsqueeze(dim=1)), dim=1)
        # print(x.shape)
        x = x.max(dim=1)[0]
        return x # init x.shape: [8, 4096]
        


def execute_exiting(func, out_ave, pre_exit_features=None):
    if pre_exit_features is not None:
        exit_door_val = func(out_ave, prev_features=pre_exit_features[-1])
    else:
        exit_door_val = func(out_ave)
    
    return exit_door_val

# fc
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, num_neurons=4096):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = [num_neurons]

        layers = []
        dim_input = input_dim
        for dim_output in self.num_neurons:
            layers.append(nn.Linear(dim_input, dim_output))
            layers.append(nn.BatchNorm1d(dim_output))
            layers.append(nn.ReLU())
            dim_input = dim_output

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def get_torchvision_model(name, pretrained=True, requires_grad=False, truncate_modules=None):
    torchvision_models = models
    # if "." in name:
    #     prefix, name = name.split(".")[0], name.split(".")[1]
    #     assert prefix in vars(torchvision_models).keys()
    #     torchvision_models = vars(torchvision_models)[prefix]
    # assert name in vars(torchvision_models).keys()

    if name == "inception_v3":
        model = vars(torchvision_models)[name](pretrained=pretrained, aux_logits=False)
    else:
        model = vars(torchvision_models)[name](pretrained=pretrained)
    if truncate_modules is not None:
        model = torch.nn.Sequential(*list(model.children())[:truncate_modules])
    for param in model.parameters():
        param.requires_grad = requires_grad

    if not requires_grad:
        model.eval()
    return model

def get_base_model(name, config, branch_idx=0):
    if name in ['vgg16', 'mobilenetv2', 'efficientnetb3', 'BN_inception', 'resnet50', 'resnet18', 'resnet34']:
        from model.branch_model import ResNet_with_Branches
        model = ResNet_with_Branches(
            branch_num=config.model.branch_num, 
            model_name=name,
            num_classes=config.model.num_classes,
            )
    else:
        raise Exception("couldn't find %s as a model name" % name)

    return model

# 判断是否退出 g(0 or 1) in the origin paper
class GateModule(nn.Module):
    def __init__(self, in_planes) -> None:
        super().__init__()
        # self.fc1 = nn.Sequential(
        #     nn.Linear(in_planes, 32),
        # )
        self.fc1 = nn.Linear(in_planes, 32)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x0, x1 = x[0], x[1]
        x0 = torch.flatten(x0, start_dim=1)
        x1 = torch.flatten(x1, start_dim=1)
        x0 = self.fc1(x0)
        x1 = self.fc1(x1)
        x = torch.cat([x0, x1], dim=1)
        return self.fc2(x)

class ExitingGate(nn.Module):
    # default in_planes = 4096
    def __init__(self, in_planes):
        super(ExitingGate, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 128, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(128, 1, bias=True)

    def forward(self, x, force_hard=True, prev_features=None):
        x0, x1 = x[0], x[1]
        
        if x0.shape[0] != 1:
            x0 = F.relu(self.bn1(self.conv1(x0)))
            x0 = F.relu(self.bn2(self.conv2(x0)))
            x1 = F.relu(self.bn1(self.conv1(x1)))
            x1 = F.relu(self.bn2(self.conv2(x1)))
        else: # when batch_size = 1, we don't use the BN layer
            x0 = F.relu(self.conv1(x0))
            x0 = F.relu(self.conv2(x0))
            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))

        x0 = torch.flatten(x0, 1)
        x1 = torch.flatten(x1, 1)
        x = torch.cat([x0, x1], dim=1)
        out = self.linear(x)
        # out = self.sigmoid(out)
        # out1 = out.clone()
        # out1[out1 >= 0.5] = 1
        # out1[out1 < 0.5] = 0
        return out

class AdaptiveBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone_type = config.model.backbone.type
        self.backbone_name = config.model.name
        self.config = config
        # base model (resnet50, EfficientNet-b3, X3D-S)
        self.backbone = get_base_model(self.backbone_name, config)
        # fully connected layer
        model_output_dim = config.model.output_dim[0]
        # num_neurons = 512
        num_neurons = config.model.output_dim[0]
        if config.data.name == 'ucf101':
            num_neurons = 4096
            self.mlp = MultiLayerPerceptron(
                input_dim=model_output_dim, num_neurons=num_neurons
            )
        self.model_output_dim = num_neurons

        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))  # in case of frame as input, the output size is 1x1, for any input size
        self.num_frames = config.data.num_frames
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, branch_idx=0,):
        b = x.size(0)
        seq_len = x.size(1)
        # Mix batch and T
        x = x.view((b * seq_len,) + x.size()[2:])
        x = self.backbone(x, branch_idx) # [8, 2048, 1, 1]
        # separate batch and T
        if len(x.shape) > 2:
            x = self.avg_pool_2d(x)  # remove spatial dim [8, 2048, 1, 1]
        
        x = x.view((b, seq_len,) + x.size()[1:]) # [8, 1, 2048, 1, 1]
        x = x.flatten(start_dim=2) # [8, 1, 2048]

        # fc layer
        x = x.view(b * seq_len, np.prod(x.size()[2:])) # [8, 2048]
        if self.config.data.name == 'ucf101':
            x = self.mlp(x) # [8, 4096]
        # x = self.dropout(x)
        x = x.view(b, seq_len, -1)

        return x

def threshold_selection(y_t, threshold=0.99):
    y_t_probs = torch.nn.Softmax(dim=1)(y_t)
    exit_door = torch.max(y_t_probs, dim=1)[0] > threshold
    return exit_door[0]

def get_policies(model, stage_num=1, show_model=False):
    params = []
    if show_model:
        for name, parameters in model.named_parameters():
            print(f"{name}: {parameters.shape}")
        return 

    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    
    # new policy for backbone and last classifier training
    if stage_num == 1.1:
        # 分支和门控不训练
        for name, parameters in model.named_parameters():
            if "branches" not in name and "exit_selector" not in name:
                params.append(parameters)
            else:
                parameters.requires_grad = False
                # print(f"{name} not in training")
    elif stage_num == 1.2:
        # 只训练分支
        for name, parameters in model.named_parameters():
            if "branches" in name:
                parameters.requires_grad = True
                # print(f"{name} in training")
                params.append(parameters)
            else:
                parameters.requires_grad = False
    elif stage_num == 1:
        # 不训练门控
        for name, parameters in model.named_parameters():
            if 'exit_selector' not in name:
                parameters.requires_grad = True
                params.append(parameters)
            else:
                parameters.requires_grad = False
                # print(f"{name} not in training")

    print("Params prepared done!")
    return params    

class TLEEModel(AdaptiveBase):
    def __init__(self, config, is_train=True, branch_num=0):
        super().__init__(config)
        self.is_train = is_train
        self.num_frames = config.data.num_frames
        self.num_classes = config.model.num_classes
        self.first_threshold = config.model.first_threshold
        self.useFeatureFunc = config.model.useFeatureFunc
        self.useGateFunc = config.model.useGateFunc
        self.exit_thresh = config.model.exit_thresh

        # featurn reuse module
        self.feature_func = None
        if self.useFeatureFunc == 'frm':
            self.FRM = FeatureReuseModule()
        elif self.useFeatureFunc == 'mp':
            self.feature_func = MaxPooling()
        elif self.useFeatureFunc == 'afrm':
            self.feature_func = AttentionFRM()
        elif self.useFeatureFunc == 'ema':
            self.feature_func = EMAFRM()
        elif self.useFeatureFunc == 'ap':
            self.feature_func = AveragePooling()
        elif self.useFeatureFunc == 'na':
            self.feature_func == NaiveAdd()

        self.exit_door = None
        self.classifiers = nn.ModuleList()
        self.classifiers.append(nn.Linear(self.model_output_dim, self.num_classes))

        if config.train_mode in [2, 'test']:
            self.exit_selector = nn.ModuleList()
            if self.useGateFunc == 'gm':
                self.exit_selector.append(GateModule(self.model_output_dim))
            elif self.useGateFunc == 'eg':
                self.exit_selector.append(ExitingGate(self.model_output_dim))
            elif self.useGateFunc == 'random':
                print("The gate module is random...")

        self.branch_num = branch_num

    def gate_selection(self, idx, y_t):
        exit_door = execute_exiting(self.exit_selector[idx], y_t)
        exit_door = nn.Sigmoid()(exit_door)
        # print(idx)
        exit_door[exit_door >= 0.5] = 1
        exit_door[exit_door < 0.5] = 0

        return False if exit_door[0] == 0 else True

    def forward(self, x, z_previous=None, t=torch.tensor(0), branch_idx=0):
        y_t = None
        # 先通过主干网络
        z_t = super().forward(x, branch_idx)
        z_t = z_t.squeeze(dim=1)
        b = z_t.shape[0]
        if t == 0:
            z_t = torch.flatten(z_t, start_dim=1)
        elif t > 0:
            # ----------------------------------------------------------------------------
            if self.feature_func is not None:
                z_t = self.feature_func(z_t, z_previous)
            else:
                z_t = torch.flatten(z_t, start_dim=1)
                
        if self.branch_num == 0: # for baseline
            if not self.is_train:
                if t == self.num_frames - 1:
                    return self.classifiers[0](z_t), None, None
                else:
                    return None, z_t, None
            

        if not self.is_train:
            '''inference mode'''
            branch_exit = False # exit from branch
            if t == 0:
                y_t = self.classifiers[0](z_t)
                exited = threshold_selection(y_t, threshold=self.first_threshold)
                # exited = self.gate_selection(t, [z_t.view(b, -1, 1, 1), z_t.view(b, -1, 1, 1)])
            elif t < self.num_frames - 1:
                if self.useGateFunc != 'random':
                    exited = self.gate_selection(
                        0, [z_t.view(b, -1, 1, 1), z_previous.view(b, -1, 1, 1)]
                    )
                else:
                    exited, branch_exit = random_exit()
            else:
                exited = True
            if self.branch_num  == 0:
                if t < self.num_frames - 1:
                    exited = False
                else:
                    exited = True
            # ------------------------------------branch_exit-----------------------------------
            if self.branch_num != 0:
                branch_exit = threshold_selection(self.classifiers[0](z_t), self.exit_thresh)

            if exited:
                if t > 0:
                    z_t = torch.flatten(z_t, start_dim=1)
                    y_t = self.classifiers[0](z_t)
                return y_t, None, branch_exit
            else:
                return None, z_t, branch_exit
        elif self.is_train:
            '''train mode'''
            y_t = self.classifiers[0](z_t)
            return y_t, z_t

        
class BaseModel(AdaptiveBase):
    def __init__(self, config):
        super().__init__(config)
        self.num_frames = config.data.num_frames
        self.num_classes = config.model.num_classes
        self.first_threshold = config.model.first_threshold
        self.max_pooling = MaxPooling()
        self.exit_door = None
        self.exited_classifiers = None
        self.classifiers = nn.Linear(self.model_output_dim, self.num_classes)
    

    def forward(self, x, z_previous=None, t=torch.tensor(0)):
        y_t = None
        # 先通过主干网络
        z_t = super().forward(x)
        z_t = z_t.squeeze(dim=1)
        if t > 0:
            z_t = self.max_pooling.forward(z_t, z_previous)
        elif t == 0:
            z_t = torch.flatten(z_t, start_dim=1)
        if t == self.num_frames - 1:
            y_t = self.classifiers(z_t)
            return y_t, z_t
        else:
            return None, z_t


if __name__ == '__main__':
    model = models.resnet50(False)
    x = torch.randn(1, 3, 224, 224)
    for _ in range(1000):
        end = time.time()
        _ = model(x)
        print(f"time: {time.time() - end:.5f}")
