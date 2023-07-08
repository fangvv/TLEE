#!/bin/bash
module load anaconda/2020.11
source activate fe

python -u main.py config/cloud_ssv2_branch.yml > log.log