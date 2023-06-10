import re
import pandas as pd
import torch
from rdkit import Chem
import sys
import random

# mask任务对应的字典
Token2Idx = {'BEG': 0, 'END': 1, 'PAD': 2, 'UNK': 3, 'MSK': 4, 'Sc': 5, 'Na': 6, 'Cr': 7, 's': 8, 'o': 9, 'Co': 10,
             'Mg': 11, 'Sm': 12, 'Si': 13, 'te': 14, 'K': 15, 'V': 16, 'C': 17, 'Ho': 18, 'Mn': 19, 'N': 20, 'W': 21,
             'Pb': 22,
             'Se': 23, 'Ti': 24, 'Ac': 25, 'Mo': 26, 'P': 27, 'In': 28, 'Dy': 29, 'La': 30, 'H': 31, 'Y': 32, 'Tc': 33,
             'Gd': 34, 'B': 35, 'Zr': 36, 'I': 37, 'Be': 38, 'Al': 39, 'Ra': 40, 'Cu': 41, 'Br': 42, 'F': 43, 'Cs': 44,
             'U': 45,
             'Ag': 46, 'Au': 47, 'Cd': 48, 'Te': 49, 'Pt': 50, 'Sn': 51, 'Tb': 52, 'c': 53, 'Ir': 54, 'Cl': 55,
             'Ru': 56,
             'Re': 57, 'Li': 58, 'Ga': 59, 'Pd': 60, 'Ge': 61, 'b': 62, 'Ba': 63, 'Yb': 64, 'Ni': 65, 'S': 66, 'Sr': 67,
             'p': 68, 'Bi': 69, 'Rh': 70, 'Hg': 71, 'O': 72, 'Eu': 73, 'As': 74, 'Ca': 75, 'Zn': 76, 'Tl': 77, 'se': 78,
             'Cf': 79, 'n': 80, 'Fe': 81, 'Nd': 82, 'Sb': 83, '/': 84, '=': 85, '+': 86, '%': 87, '#': 88, '[': 89,
             ']': 90,
             '\\': 91, '-': 92, '@': 93, '.': 94, '(': 95, '*': 96, ')': 97, '$': 98, ':': 99, '0': 100, '1': 101,
             '2': 102,
             '3': 103, '4': 104, '5': 105, '6': 106, '7': 107, '8': 108, '9': 109, '@@': 110, '+1': 111, '-1': 112,
             '+2': 113,
             '-2': 114, '+3': 115, '-3': 116, '+4': 117, '-4': 118, '+5': 119, '-5': 120, '+6': 121, '-6': 122,
             '+7': 123,
             '-7': 124, '+8': 125, '-8': 126, '%10': 127, '%11': 128, '%12': 129, '%13': 130, '%14': 131, '%15': 132,
             '%16': 133, '%17': 134, '%18': 135, '%19': 136}


def smi_tokenizer(smi, max_len, padding=True, mask_prob=0):
    """
    按照rdkit提取的原子信息解析SMILES串
    padding：是否固定长度提取，padding=True时，固定返回长度为max len的列表
    """
    smi = smi.strip()

    Mol = Chem.MolFromSmiles(smi)

    atom_list = [atom.GetSymbol() for atom in Mol.GetAtoms()]

    now = 0
    tokens = []
    i = 0
    while i < len(smi):
        # 匹配原子
        if now < len(atom_list):
            atom = atom_list[now]
            if len(atom) == 1 and smi[i].upper() == atom.upper():
                tokens.append(smi[i])
                now += 1
                i += 1
                continue
            if len(atom) == 2 and smi[i:i + 2].upper() == atom.upper():
                tokens.append(smi[i:i + 2])
                now += 1
                i += 2
                continue
        # 匹配+num / -num
        if (smi[i] == '+' or smi[i] == '-') and smi[i + 1] in "1234567890":
            tokens.append(smi[i: i + 2])
            i += 2
            continue
        # 匹配 %num (编号大于等于10的环)
        if smi[i] == '%':
            tokens.append(smi[i: i + 3])
            i += 3
            continue
        # 匹配 @@ (手性异构)
        if smi[i] == '@' and smi[i + 1] == '@':
            tokens.append(smi[i: i + 2])
            i += 2
            continue
        # 匹配单个符号
        tokens.append(smi[i])
        i += 1

    # print(atom_list)
    # print(''.join(tokens))
    # print(smi)
    # exit(0)
    # print(tokens)

    # for token in tokens:
    #     if token not in Token2Idx.keys():
    #         print(f"{token} is not in keys!!")

    # if smi != ''.join(tokens):
    #     print("部分元素未被提取")
    #     print(smi)
    #     print(atom_list)
    #     print(''.join(tokens))

    if mask_prob != 0:
        tokens = ['MSK' if random.random() < mask_prob else token for token in tokens]

    if padding:
        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len // 2 - 1] + tokens[-max_len // 2 + 1:]
        content = [Token2Idx.get(token, Token2Idx['UNK']) for token in tokens]
        # print(content)
        X = [Token2Idx['BEG']] + content + [Token2Idx['END']]
        padding = [Token2Idx['PAD']] * (max_len - len(X))
        X.extend(padding)
        return X
    else:
        content = [Token2Idx.get(token, Token2Idx['UNK']) for token in tokens]
        X = [Token2Idx['BEG']] + content + [Token2Idx['END']]
        return X


NewDic = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<mask>': 4, ':': 5, '[': 6, ']': 7, '2': 8, '1': 9, 'H': 10,
          'c': 11, 'C': 12, '3': 13, '(': 14, ')': 15, '4': 16, '5': 17, 'O': 18, '6': 19, '7': 20, '8': 21, '9': 22,
          '0': 23, '=': 24, 'N': 25, 'n': 26, 'F': 27, '-': 28, 'S': 29, '/': 30, 'Cl': 31, 's': 32, 'o': 33, '#': 34,
          '+': 35, 'Br': 36, '\\': 37, 'P': 38, 'I': 39, '-2': 40, '-3': 41, 'Si': 42, 'B': 43, '-4': 44}


def split(sm):
    """
    function: Split SMILES into words. Care for Cl, Br, Si, Se, Na etc.
    input: A SMILES
    output: A string with space between words
    """
    arr = []
    i = 0
    while i < len(sm) - 1:
        if not sm[i] in ['%', 'C', 'B', 'S', 'N', 'R', 'X', 'L', 'A', 'M', 'T', 'Z', 's', 't', 'H', '+', '-', 'K', 'F']:
            arr.append(sm[i])
            i += 1
        elif sm[i] == '%':
            arr.append(sm[i:i + 3])
            i += 3
        elif sm[i] == 'C' and sm[i + 1] == 'l':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'C' and sm[i + 1] == 'a':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'C' and sm[i + 1] == 'u':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'B' and sm[i + 1] == 'r':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'B' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'B' and sm[i + 1] == 'a':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'B' and sm[i + 1] == 'i':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'S' and sm[i + 1] == 'i':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'S' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'S' and sm[i + 1] == 'r':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'N' and sm[i + 1] == 'a':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'N' and sm[i + 1] == 'i':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'R' and sm[i + 1] == 'b':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'R' and sm[i + 1] == 'a':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'X' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'L' and sm[i + 1] == 'i':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'A' and sm[i + 1] == 'l':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'A' and sm[i + 1] == 's':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'A' and sm[i + 1] == 'g':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'A' and sm[i + 1] == 'u':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'M' and sm[i + 1] == 'g':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'M' and sm[i + 1] == 'n':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'T' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'Z' and sm[i + 1] == 'n':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 's' and sm[i + 1] == 'i':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 's' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 't' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'H' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '+' and sm[i + 1] == '2':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '+' and sm[i + 1] == '3':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '+' and sm[i + 1] == '4':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '-' and sm[i + 1] == '2':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '-' and sm[i + 1] == '3':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '-' and sm[i + 1] == '4':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'K' and sm[i + 1] == 'r':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'F' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        else:
            arr.append(sm[i])
            i += 1
    if i == len(sm) - 1:
        arr.append(sm[i])
    return ' '.join(arr)


def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm) > 218:
        # print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109] + sm[-109:]
    ids = [NewDic.get(token, NewDic['<unk>']) for token in sm]
    ids = [NewDic['<sos>']] + ids + [NewDic['<eos>']]
    seg = [1] * len(ids)
    padding = [NewDic['<pad>']] * (seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg


def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a, b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)


if __name__ == '__main__':
    # log_file = open("message.log", "w")
    # sys.stdout = log_file
    file = "/home/chembl_27.csv"
    smiles = pd.read_csv(file)['smiles'].values[:10]
    for smi in smiles:
        # print(smi)
        tokens = smi_tokenizer(smi, max_len=440, padding=True)
        print(tokens)
        mask_tokens = smi_tokenizer(smi, max_len=440, padding=True, mask_prob=0.15)
        print(mask_tokens)
        print('\n')

    # sme = SmilesEnumerator()
    # for smi in smiles:
    #     try:
    #         sme.randomize_smiles(smi)
    #     except Exception:
    #         print(smi)
