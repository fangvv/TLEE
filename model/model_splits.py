import sys
sys.path.append("..")

import torch
import torch.nn as nn

def get_model_structure(model_name='vgg16', output_dim=0):
    pool_layer = None
    fc_layer = None
    if model_name == 'vgg16':
        from arch.VGG16 import model
        pool_layer = nn.Sequential(
            model.avgpool,
        ).cuda()
        fc_layer = nn.Sequential(
            *list(model.classifier.children())[:-2],
            nn.Linear(4096, 512),
        ).cuda()
        
        model = model.features.cuda()
        split_location = [14, 24, 34, -1]
        split_map_size = [56, 28, 14, 7]
        split_channels = [128, 256, 512, 512]
        # res_parts = get_res_parts(model.features, split_location, split_map_size, split_channels)
        # avgpool_layer = model.avgpool
        # res_parts.append(avgpool_layer) # 4 + 1
        # get_map_size(res_parts)
    elif model_name == 'BN_Inception':
        # not use
        from arch.BN_Inception import bninception
        model = bninception()
        split_location = ['_block_1', '_block_3a', '_block_4a', '_block_5a']
        split_map_size = [28, 14, 7, 1]
        split_channels = [192, 576, 1056, 1024]
        # res_parts = get_inception_res_parts(model, split_location, split_map_size, split_channels)
        # get_increption_map_size(res_parts)
    elif model_name == 'mobilenetv2':
        # 最后一层针对H和W不为1,采用x.mean(3).mean(2)
        from arch.mobilenetv2 import model
        model = torch.nn.Sequential(*list(model.features.children()))
        # _, _ = get_model_complexity_info(model, (3, 224 ,224))
        split_location = [4, 8, 12, -1]
        split_map_size = [56, 14, 14, 7]
        split_channels = [24, 64, 96, 1280]
        # res_parts = get_res_parts(model.features, split_location, split_map_size, split_channels)
        # get_map_size(res_parts)
    elif model_name == 'efficientnetb3':
        from arch.efficientnet import model
        # pool_layer = nn.Sequential(
        #     model.avgpool,
        # ).cuda()
        # fc_layer = nn.Sequential(
        #     model.classifier[0],
        # ).cuda()
        model = model.features.cuda()
        split_location = [2, 4, 6, -1]
        split_map_size = [112, 28, 14, 7]
        split_channels = [24, 48, 136, 1536]
        # res_parts = get_res_parts(model.features, split_location, split_map_size, split_channels)
        # avgpool_layer = model.avgpool
        # res_parts.append(avgpool_layer) # 4 + 1
        # get_map_size(res_parts)
    return split_location, split_map_size, split_channels, model, pool_layer, fc_layer

def get_res_parts(model, split_location=[], split_map_size=[], split_channels=[]):
    res_parts = []
    n = len(split_location)
    model = nn.Sequential(*list(model.children()))
    for i in range(n):
        if i == 0:
            part = model[:split_location[i]]
        elif i > 0 and i < n - 1:
            part = model[split_location[i - 1]:split_location[i]]
        else:
            part = model[split_location[i - 1]:]
        res_parts.append(part)
    return res_parts

def get_map_size(res_parts):
    n = len(res_parts)
    for i in range(n):
        x = torch.randn(1, 3, 224, 224)
        for j in range(i + 1):
            x = res_parts[j](x)
        print(f"{i} {x.shape}")

def get_inception_res_parts(model, split_location=[], split_map_size=[], split_channels=[]):
    res_parts = []
    n = len(split_location)
    for i in range(n):
        part = []
        if i == 0:
            part.append(model._block_1)
            part.append(model._block_2)
        elif i == 1:
            part.append(model._block_3a)
            part.append(model._block_3b)
            part.append(model._block_3c)
        elif i == 2:
            part.append(model._block_4a)
            part.append(model._block_4b)
            part.append(model._block_4c)
            part.append(model._block_4d)
            part.append(model._block_4e)
        elif i == 3:
            part.append(model._block_5a)
            part.append(model._block_5b)
            part.append(model.global_pool)
        res_parts.append(part)
        
    return res_parts

def get_increption_map_size(res_parts):
    for i in range(len(res_parts)):
        x = torch.randn(1, 3, 224, 224)
        for j in range(i + 1):
            for block in res_parts[j]:
                x = block(x)
        print(f"{i} {x.shape}")

if __name__ == '__main__':
    # get_model_structure('vgg16')
    # get_model_structure('BN_Inception')
    get_model_structure('mobilenetv2')
    # get_model_structure('efficientnet_b3')
    print("Done!")