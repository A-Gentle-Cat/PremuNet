from .pretrain_trfm import TrfmSeq2seq
from .utils import smi_tokenizer, split
from .utils import Token2Idx, NewDic
import pandas as pd
import torch

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4

trfm = TrfmSeq2seq(len(NewDic), 256, len(NewDic), 4)
trfm.load_state_dict(torch.load('./dataset/1DFeature/trfm_12_23000.pkl'))
print("Transformer权重读入完成")
trfm.eval()


def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm) > 218:
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109] + sm[-109:]
    ids = [NewDic.get(token, NewDic['<unk>']) for token in sm]
    ids = [sos_index] + ids + [eos_index]
    seg = [1] * len(ids)
    padding = [pad_index] * (seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg


def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a, b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)


def Tsfm_fp(smiles):
    """
    smiles: 长度为N的SMILES list
    return X: [N, 1024] 各SMILES对应的tsfm指纹
    """
    x_split = [split(sm) for sm in smiles]
    xid, _ = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X




if __name__ == '__main__':
    df = pd.read_csv('/tmp/pycharm_project_747/dataset/bbbp.csv')
    smiles = df['smiles'].values
    X = Tsfm_fp(smiles)
    print(X.shape)
