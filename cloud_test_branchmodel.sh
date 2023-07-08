#!/bin/bash
module load anaconda/2020.11
source activate fe

python -u model/branch_model.py config/cloud_ucf101_branch.yml > log.log