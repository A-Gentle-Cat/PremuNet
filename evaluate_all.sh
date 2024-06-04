#!/bin/sh

# change this !
your_python_path=/usr/local/bin/python
your_dataset_directory=/root/dataset

for dataset in BBBP BACE clintox sider ESOL Freesolv Lipophilicity TOX21 ;
do
  ${your_python_path} train.py --dataset ${dataset} --model PremuNet --dataset_dir ${your_dataset_directory}
done
