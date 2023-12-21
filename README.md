# A Pretrained Multi-Representation Network for Molecular Property Prediction
### [Paper](https://www.sciencedirect.com/science/article/pii/S1566253523004086) | [Github](https://github.com/A-Gentle-Cat/PremuNet) 

## OverView

PremuNet can extract information from multiple views and combine them interactively through pre-training and ﬁne-tuning. The network consists of two branches: a Transformer-GNN branch that extracts SMILES and graph information, and a Fusion Net branch that extracts graph and conformation information, called PermuNet-L and PermuNet-H respectively. We employ masked self-supervised methods to enable the model to learn information fusion and achieve improved performance in downstream tasks.
<div align="center">
<img width="2500" src="DrawGraph/totalmodel.png" alt="totalmodel" style="zoom:14%;" />
</div>

## Dataset
### Dataset for pretraining.
Our pre-training process employs both the ChEMBL and PCQM4M datasets. The ChEMBL dataset can be accessed through [this link](https://chembl.gitbook.io/chembl-interface-documentation/downloads), while the PCQM4M dataset is available for download [here](http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz).
### Dataset for finetune
We use MoleculeNet as our benchmark test. The datasets and splits of MoleculeNet in our experiment are available at [this link](http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/). Download and unzip it into `/dataset`.

We have added 3D coordinates to the molecules in MoleculeNet to further improve prediction performance. The source for these 3D coordinates can be accessed via [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF). 
You should download the `molecule_net.tar.gz` file. Unzip it and place it in the `/dataset` directory
## Setup Environment
```shell
pip install -r requirements.txt
```
## Pretrain
The pretraining weights can be found at [this link](https://drive.google.com/drive/folders/16ANfFYW4uKYrBzn56v94hkAGu4yGg7tM?usp=share_link).
Please download them and put them in the `/model_weight` directory.
If you want to train your own pre-training weights, you can refer to the following steps.
### SMILES-Transformer
#### For Atom-Level Pretrain
To pretrain the Atome-Level Transformer, you can follow these steps:
```shell
cd Transformer1D
python pretrain_trfm.py
```
Please note that you should prepare your own dataset for pre-training and modify parts of the code.
#### For Graph-Level Pretrain
The pre-training of Graph-Level Transformer can follow the introduction in [this link](https://github.com/DSPsleeporg/smiles-transformer).
### PremuNet-H
PremuNet-H module, we used the code from [this link](https://github.com/teslacool/UnifiedMolPretrain).
You can consider using their code and steps for pre-training, or use our code with the following steps:
```shell
cd UnifiedMolPretrain
python train.py --num-layers 12 --batch-size 128 --enable-tb --node-attn --use-bn --pred-pos-residual --mask-prob 0.25
```
## Finetuning
The configuration files for each dataset are located in the /configs directory. These parameters can be utilized directly as provided, or they can be modified as needed. The step for finetuning on MoleculeNet is as follows
### Step 1: Download Dataset
You should unzip the downloaded dataset and place them separately under the `/dataset/` directory.
### Step 2: Get Molecular Feature
```shell
python pretrans_data.py
```
### Step 3: Start finetuning
```shell
bash evaluate_all.sh
```

## Cite
Zhang H, Wu J, Liu S, et al. A pre-trained multi-representation fusion network for molecular property prediction[J]. Information Fusion, 2024, 103: 102092.
## Reference
This project utilizes the code from the [UnifiedMolPretrain](https://github.com/teslacool/UnifiedMolPretrain) and [Smiles-Transformer](https://github.com/DSPsleeporg/smiles-transformer) projects. The successful implementation of this project is attributed to the outstanding research papers and well-structured code provided by these initiatives.
