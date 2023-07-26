import torch
from FBG_3DGen.model import * 
from FBG_3DGen.comparm import * 
import argparse as arg
import pickle,os
from rdkit.Chem import Draw
from rdkit import Chem
import numpy as np 
from tqdm import tqdm 
torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=20)
parser=arg.ArgumentParser(description='Graph-CRealNVP')
parser.add_argument('-i','--input')
parser.add_argument('--gpu')
args=parser.parse_args()
jsonfile=args.input
gpu=args.gpu
UpdateGPARAMS(jsonfile)
os.environ["CUDA_VISIBLE_DEVICES"] =gpu
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

molgen=MolGen_Model_RL(prior_modelname='MFTs_Gen_Model',loadtype="Perepoch")
total_mols=[]
total_smis=[]
os.system(f'mkdir -p ./sampled_mols')
with open('sampled_mols.smi','w') as f:
    for i in tqdm(range(1)):
        molnodes,moledges,molatomnums,mol_likelihoods_agent,mol_likelihoods_prior=molgen.sample_batch()
        mols,smis,vids=molgen.nodes_edges_to_mol(molnodes,moledges,molatomnums)
        total_mols+=mols
        total_smis+=smis
        img=Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(250,250))
        img.save(f'./sampled_mols/sampled_mols_{i}.png')
        for smi in smis:
            f.write(f'{smi}\n')

