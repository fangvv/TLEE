'''
TODO
    为主干网络添加一个辅助分类器用来判断是否从当前分支退出
'''

from email.policy import strict
from multiprocessing import pool
import numbers
import sys
sys.path.append(".")
import time

import torch
import torch.nn as nn
import torchvision.models as models

from read_config import config
from log._log import logging
from model.model_splits import get_model_structure
from model.branch import Conv_Branch, FC_Branch


model_name = config.model.backbone
class Branch(nn.Module):
    def __init__(self, in_planes, channels, num_classes=101, alpha=0.5) -> None:

        super().__init__()
        if in_planes >= 4:
            kernel_size = in_planes // 4
        else:
            kernel_size = -1
        scale = config.model.branch_scale

        # Channel Reduction Block
        crm_num = config.model.crm_num
        self.crm = nn.Sequential()
        last_scale = 2 ** scale
        for i in range(crm_num):
            if i == 0:
                self.crm.add_module(
                    f'conv{i}', nn.Conv2d(channels, channels // last_scale, kernel_size=1),
                )
            else:
                self.crm.add_module(
                    f"conv{i}", nn.Conv2d(channels // last_scale, channels // (last_scale * 2), kernel_size=1)
                )
                last_scale *= 2
            self.crm.add_module(
                f'relu{i}', nn.ReLU(inplace=True),
            )
        output_dim = config.model.output_dim[0]
        self.fc1 = nn.Linear(kernel_size * kernel_size * (channels // last_scale), output_dim)
        self.forward = self._forward_wo_fc

        if kernel_size == -1:
            # base
            # self.fc1 = nn.Linear(in_planes * in_planes * channels, num_classes)
            self.forward = self._forward_wo_fc
        else:
            kernel_size_1 = in_planes // kernel_size
            self.max_pool = nn.MaxPool2d(kernel_size=kernel_size_1)
            self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size_1)
            self.alpha = nn.Parameter(torch.rand(1))
            # base
           
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _forward_wo_fc(self, x):
        '''
        不通过fc层
        '''
        # print(f"x.shape: {x.shape}")
        mixed = self._base_process_with_avg_max_pooling(x)
        # print(f"mixed.shape: {mixed.shape}")
        x = self.crm(mixed)
        x = self.fc1(x.view(x.size(0), -1))
        # x = self.relu(x)
        return x
    
    def _base_process_with_avg_max_pooling(self, x):
        '''
        混合平均最大池化
        '''
        avgp = self.alpha * self.max_pool(x)
        maxp = (1 - self.alpha) * self.avg_pool(x)
        mixed = avgp + maxp
        return mixed

    def _forward_w_pooling(self, x):
        '''
        池化并通过fc层
        '''
        mixed = self._base_process_with_avg_max_pooling(x)
        return self.fc1(mixed.view(mixed.size(0), -1))

    def _forward_wo_pooling(self, x):
        '''
        不池化通过fc层
        '''
        return self.fc1(x.view(x.shape[0], -1))

def get_ResNet(model_name='resnet50', pretrained=True, num_classes=101):

    logging.info(f'Getting model {model_name} ...')
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(2048, num_classes)
        return model
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(512, num_classes)
        return model
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(512, num_classes)
        return model
    else:
        raise NotImplementedError(f"Unknown Model Name {model_name}, please check!")

class ResNet_with_Branches(nn.Module):
    def __init__(self, branch_num=0, model_name='resnet50', num_classes=101) -> None:
        super().__init__()
        self.model_name = model_name
        self.baseModel = None
        if 'resnet' in model_name:
            self.baseModel = get_ResNet(self.model_name, pretrained=False, num_classes=num_classes)
            self.baseModel = torch.nn.Sequential(*list(self.baseModel.children())[:-1])
        self.inter_feature = None
        self.branches = nn.ModuleList()
        self.branch_num = branch_num
        self.num_classes = num_classes
        self.res_part = self._get_splits(model_name, self.baseModel)
        self.forward = self._forward_all_branch
        


    def _get_splits(self, model_name, model=None) -> list:
        # 各个分支之前的主网络
        _model = nn.Sequential(*list(model.children())) if 'resnet' in model_name else None
        res_part = []
        self.fc_layer = None
        self.pool_layer = None
        if model_name == 'resnet50':
            # !!!只适用于resnet50
            split_location = [5, 6, 7, -1]
            split_map_size = [56, 28, 14, 7] # 224x224 [56, 28, 14] (ucf101) 32x32 [8, 4, 2](cifar10)
            split_channels = [256, 512, 1024, 2048]
        elif model_name == 'resnet18':
            split_location = [5, 6, 7, -1]
            split_map_size = [56, 28, 14, 7]
            split_channels = [64, 128, 256, 512]
        elif model_name in ['vgg16', 'mobilenetv2', 'efficientnetb3']:
            split_location, split_map_size, split_channels, _model, pool_layer, fc_layer = get_model_structure(model_name, config.model.output_dim[0])
            # post process
            self.pool_layer = pool_layer
            self.fc_layer = fc_layer
            self.baseModel = _model
        else:
            raise NotImplementedError(f"Not implement this model {model_name}, please change to resnet50")
        
        split_location = split_location[-(self.branch_num + 1):]
        for i in range(self.branch_num + 1):
            # 存储由分支隔开的每一段主网络
            if self.branch_num == 0: # 没有分支
                res_part += [_model]
            elif i == 0 and self.branch_num != 0:
                res_part += [_model[:split_location[i]]]
            elif i < self.branch_num:
                res_part += [_model[split_location[i - 1]:split_location[i]]]
            else:
                res_part += [_model[split_location[i - 1]:]]
            # 存储分支
            if i != self.branch_num:
                self.branches.append(
                    Branch(
                        in_planes=split_map_size[-(self.branch_num + 1):][i], 
                        channels=split_channels[-(self.branch_num + 1):][i], 
                        num_classes=self.num_classes, 
                        alpha=0.5
                        )
                    # Conv_Branch(
                    #     size=split_map_size[-(self.branch_num + 1):][i],
                    #     inplanes=split_channels[-(self.branch_num + 1):][i],
                    #     outputdim=config.model.output_dim[0]
                    # )
                    # FC_Branch(
                    #     size=split_map_size[-(self.branch_num + 1):][i],
                    #     inplanes=split_channels[-(self.branch_num + 1):][i],
                    #     outputdim=config.model.output_dim[0]
                    # )
                    )
        return res_part


    def _forward_i_branch(self, i, x) -> list:
        '''只训练第i个分支'''
        x = self.res_part[i](x)
        x = self.branches[i](x)
        x = self.baseModel.fc(x)
        return [x]  
           
    def _forward_all_branch(self, x, branch_idx):
        # non-branch model or train backbone without training branch
        if self.branch_num == 0 or (self.inter_feature == None and branch_idx == self.branch_num):
            for part in self.res_part:
                x = part(x)
            if self.pool_layer is not None:
                x = self.pool_layer(x)
            if self.fc_layer is not None:
                x = self.fc_layer(x.view(x.size(0), -1))
            return x
        # branch model
        elif 0 <= branch_idx < self.branch_num:
            if branch_idx == 0:
                self.inter_feature = None
            if self.inter_feature == None:
                x = self.res_part[branch_idx](x)
            else:
                x = self.res_part[branch_idx](self.inter_feature)
            self.inter_feature = x
            x = self.branches[branch_idx](x)
            return x
        elif branch_idx == self.branch_num or branch_idx == -1: # last classifier
            x = self.res_part[branch_idx](self.inter_feature)
            if self.pool_layer is not None:
                x = self.pool_layer(x)
            if self.fc_layer is not None:
                x = self.fc_layer(x.view(x.size(0), -1))
            self.inter_feature = None # model inference complete, so set inter_feature to None
            return x
        else:
            raise NotImplementedError(f"Branch index {branch_idx} out of max branch number {self.branch_num}")

def unit_test():
    # getResNet('resnet50', True)
    end = time.time()
    model = ResNet_with_Branches(2, model_name='mobilenetv2')
    x = torch.randn(1, 3, 224, 224)
    # print(model)
    with torch.no_grad():
        print("Branch")
        for i in range(3):
            model(x, i)
        
    print("Done!")

if __name__ == '__main__':
    unit_test()