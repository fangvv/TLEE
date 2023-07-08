#!/bin/bash

module load anaconda/2020.11
source activate fe

python -u test.py > log.log