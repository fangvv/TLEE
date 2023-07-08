#!/bin/bash
module load anaconda/2020.11
source activate fe

# resnet50
python -u main.py config/cloud_hmdb51_resnet50_branch.yml > log.log

# # mobilenetv2
# python -u main.py config/cloud_hmdb51_mobilenetv2_branch.yml > log.log

# efficientnetb3
# python -u main.py config/cloud_hmdb51_efficientnetb3_branch.yml > log.log

# vgg16
# python -u main.py config/cloud_hmdb51_vgg16_branch.yml > log.log