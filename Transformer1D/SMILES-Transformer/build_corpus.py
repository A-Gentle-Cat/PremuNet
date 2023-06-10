from rdkit import Chem
from tqdm import tqdm
from utils.DataReader import BBBP_reader


def build_corpus():
    file_path = "/Users/rune/PycharmProjects/Transformer1D/dataset/chembl_v27_standardized.sdf"
    out_path = "/Users/rune/PycharmProjects/Transformer1D/TransformerData/chembl_corpus.txt"

    Mols = Chem.SDMolSupplier(file_path)
    smiles = [Chem.MolToSmiles(mol) for mol in Mols]

    print("smiles串的个数：", len(smiles))

    with open(out_path, 'a') as f:
        for sm in tqdm(smiles):
            f.write(split(sm) + '\n')
    print("Built a corpus file!")


def Get_Atom():
    file_path = "/Users/rune/PycharmProjects/Transformer1D/dataset/chembl_v27_standardized.sdf"
    out_path = "/Users/rune/PycharmProjects/Transformer1D/TransformerData/chembl_corpus.txt"

    # Mols = Chem.SDMolSupplier(file_path)
    # st = set()
    # for mol in tqdm(Mols):
    #     atoms = mol.GetAtoms()
    #     for atom in atoms:
    #         st.add(Chem.Atom.GetSymbol(atom))
    # print(st)
    st = set()
    smiles, labels = BBBP_reader()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        atoms = mol.GetAtoms()
        for atom in atoms:
            st.add(Chem.Atom.GetSymbol(atom))
    print(st)

# Split SMILES into words
def split(sm):
    """
    function: Split SMILES into words. Care for Cl, Br, Si, Se, Na etc.
    input: A SMILES
    output: A string with space between words
    """
    arr = []
    i = 0
    while i < len(sm) - 1:
        if not sm[i] in ['%', 'C', 'B', 'S', 'N', 'R', 'X', 'L', 'A', 'M',
                         'T', 'Z', 's', 't', 'H', '+', '-', 'K', 'F']:
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


if __name__ == '__main__':
    Get_Atom()
