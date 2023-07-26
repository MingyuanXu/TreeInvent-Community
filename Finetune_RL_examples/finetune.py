import torch
from FBG_3DGen.model import * 
from FBG_3DGen.comparm import * 
from FBG_3DGen.graphs import *
import argparse as arg
import pickle,os
from rdkit import Chem
from tqdm import tqdm 
parser=arg.ArgumentParser(description='Graph-CRealNVP')
parser.add_argument('-i','--input')
parser.add_argument('--gpu')
args=parser.parse_args()
jsonfile=args.input
gpu=args.gpu
UpdateGPARAMS(jsonfile)
os.environ["CUDA_VISIBLE_DEVICES"] =gpu
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

if GP.ftsetting.smi_path!='':
    smis=Load_smiles_list(GP.ftsetting.smi_path)
    mols=prepare_mols_from_smi(smis)
if GP.ftsetting.sdf_path!='':
    mols=prepare_mols_from_sdf(GP.ftsetting.sdf_path)
for i in range(517,518):
    ftmols=[]
    for mol in mols:
        similarity=tanimoto_similarities(mol,mols[i])   
        if similarity>0.7:
            ftmols.append(mol)
    print (len(ftmols))
ft_smis=[Chem.MolToSmiles(mol) for mol in ftmols]
Save_smiles_list(ft_smis,'ft_datasets.smi')
npages=math.ceil(len(ftmols)/100)
for i in range(npages):
    img=Draw.MolsToGridImage(ftmols[i*100:(i+1)*100],molsPerRow=5,subImgSize=(250,250))
    img.save(f'datasets_p{i}.png')
ft_MFTs=create_finetune_datasets(ftmols*5,cut=0.97)
#print (ft_MFTs)

molgen=MolGen_Model(modelname='MFTs_Gen_Model',loadtype='Perepoch')
molgen.epochs=0
molgen.finetune(ft_MFTs,epochnum=5000,)

