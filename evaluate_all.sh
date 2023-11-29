#!/bin/sh

# change this !
your_python_path=/usr/local/miniconda3/bin/python

for dataset in BBBP BACE clintox sider ESOL Freesolv Lipophilicity TOX21 ;
do
  ${your_python_path} train.py --dataset ${dataset} --model PremuNet --print_to_log --agg_method concat
done
