#!/bin/bash

python scripts/generate_data.py data/raw/minimass --object-set minimass --num-grasps 50000
python scripts/clean_data data/raw/minimass
python scripts/construct_dataset/py data/raw/minimass data/datasets/minimass 
python scripts/train_vgn.py --dataset /data/datasets/minimass --epochs 30 --batch-size 32