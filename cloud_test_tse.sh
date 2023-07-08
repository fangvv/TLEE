#!/bin/bash
module load anaconda/2020.11
source activate fe

python -u model/tse.py config/cloud_ucf101_resnet50_branch.yml> log.log