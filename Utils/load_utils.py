import os.path

from sklearn.model_selection import train_test_split

from deepchem.splits.splitters import *
import config


def load_smiles_and_label(dataset_name, item):
    smiles, label = None, None
    if dataset_name == 'BBBP':
        smiles, label = item[1], [item[0]]
    if dataset_name == 'BACE':
        smiles, label = item[1], [item[0]]
    if dataset_name == 'clintox':
        smiles, label = item[2], [item[0], item[1]]
    if dataset_name == 'HIV':
        smiles, label = item[1], [item[0]]
    if dataset_name == 'TOX21':
        smiles, label = item[12], [label for label in item[0:12]]
    if dataset_name == 'sider':
        smiles, label = item[27], [label for label in item[0:27]]
    if dataset_name == 'QM9':
        smiles = item[1]
        label = [cur for cur in item[2:]]
        return smiles, label
    if dataset_name == 'TOXCAST':
        smiles, label = item[617], [label for label in item[0:617]]
    if dataset_name == 'ESOL':
        smiles, label = item[1], item[0]
    if dataset_name == 'Freesolv':
        smiles, label = item[1], item[0]
    if dataset_name == 'Lipophilicity':
        smiles, label = item[1], item[0]
    if smiles is None:
        raise Exception('没有这个数据集！', dataset_name)
    return smiles, label



def get_task_num(dataset_name) -> int:
    if dataset_name == 'BBBP':
        return 1
    if dataset_name == 'BACE':
        return 1
    if dataset_name == 'clintox':
        return 2
    if dataset_name == 'HIV':
        return 1
    if dataset_name == 'TOX21':
        return 12
    if dataset_name == 'sider':
        return 27
    if dataset_name == 'QM9':
        return 19
    if dataset_name == 'TOXCAST':
        return 617
    if dataset_name == 'Freesolv':
        return 1
    if dataset_name == 'Lipophilicity':
        return 1
    if dataset_name == 'ESOL':
        return 1
    raise Exception('没有这个数据集！')

def get_dataset_type(dataset_name):
    if dataset_name in ['BBBP', 'BACE', 'clintox', 'HIV', 'TOX21', 'sider', 'TOXCAST']:
        return 'classification'
    elif dataset_name in ['QM9', 'QM8', 'QM7']:
        return 'regression'
    else:
        raise Exception('没有这个数据集！')

def get_lineandgraph_save_names(dataset_name, catogory, process_dir):
    path1 = []
    path2 = []
    path3 = []
    paths = ['data_train', 'counter_train',
             'data_valid', 'counter_valid',
             'data_test', 'counter_test',
             'data', 'counter']
    if catogory == 'train':
        paths = paths[0:2]
    elif catogory == 'valid':
        paths = paths[2:4]
    elif catogory == 'test':
        paths = paths[4:6]
    else:
        paths = paths[6:8]

    for i, path in enumerate(paths):
        path1.append(path + '.pt')
    for i, path in enumerate(paths):
        path2.append(path + '_lineGraph.pt')
    for i, path in enumerate(paths):
        path2.append(path + '_fingerprints.pt')

    res = path1 + path2 + path3
    for i, path in enumerate(res):
        res[i] = os.path.join(process_dir, path)

    return res

def get_dataset_spliter(dataset_name):
    if dataset_name in ['BBBP', 'HIV', 'BACE']:
        return ScaffoldSplitter()
    elif dataset_name in ['PDBbind', 'QM9']:
        return RandomSplitter()
    else:
        return RandomStratifiedSplitter()


def get_split_idx(data_dir, data_length):
    if config.split_type == 'scaffold':
        data_dir = os.path.join(data_dir, 'split', 'scaffold')
        train_idx = pd.read_csv(os.path.join(data_dir, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
        valid_idx = pd.read_csv(os.path.join(data_dir, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
        test_idx = pd.read_csv(os.path.join(data_dir, 'test.csv.gz'), compression='gzip', header=None).values.T[0]
    elif config.split_type == 'random':
        train_idx, valid_idx = train_test_split(range(0, data_length), test_size=0.2, shuffle=True)
        valid_idx, test_idx = train_test_split(valid_idx, test_size=0.5, shuffle=True)
    else:
        raise Exception('split type ERROR!')

    return {'train': np.array(train_idx), 'valid': np.array(valid_idx), 'test': np.array(test_idx), 'all': np.array(range(0, len(train_idx)+len(valid_idx)+len(test_idx)))}
