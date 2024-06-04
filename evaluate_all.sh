#!/bin/sh

# change this !
your_python_path=/usr/local/bin/python

for dataset in BACE;
do
  ${your_python_path} train.py --dataset ${dataset} --model PremuNet --dataset_dir /root/dataset
done
