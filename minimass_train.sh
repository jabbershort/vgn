#!/bin/bash

datasetname=minimass

python scripts/generate_data.py data/raw/$datasetname --object-set $datasetname --num-grasps 50000
python scripts/clean_data.py data/raw/$datasetname
python scripts/construct_dataset.py data/raw/$datasetname data/datasets/$datasetname
python scripts/train_vgn.py --dataset data/datasets/$datasetname --epochs 60 --batch-size 64 --description $datasetname
