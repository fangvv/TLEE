#!/bin/bash
module load anaconda/2020.11
source activate fe

# python -u train_basemodel.py config/cloud_hmdb51_resnet50_branch.yml > log.log

# python -u train_basemodel.py config/cloud_ucf101_mobilenetv2_branch.yml > log.log

# vgg
# hmdb
python -u train_basemodel.py config/cloud_hmdb51_vgg16_branch.yml > log.log
