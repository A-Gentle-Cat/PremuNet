#!/bin/sh

your_python_path=/usr/local/miniconda3/envs/gnn/bin/python

for dataset in BBBP BACE clintox sider ESOL Freesolv Lipophilicity TOX21 ;
do
  ${your_python_path} train.py --dataset ${dataset} --model PremuNet --print_to_log --agg_method concat
done
