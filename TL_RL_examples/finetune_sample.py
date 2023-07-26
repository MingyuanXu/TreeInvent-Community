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

target_smis=Load_smiles_list('ft_datasets.smi')
target_fps=np.array([AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useCounts=True, useFeatures=True) for smi in target_smis])
total_mols=[]
total_smis=[]
picpath='./FT_mols'
os.system(f"mkdir -p {picpath}")
molgen=MolGen_Model(modelname='MFTs_Gen_Model',loadtype='Finetune')
for temp in tqdm(np.arange(1,5.01,0.05)):
    mols,smis,validity=molgen.sample(sample_num=128,temp=temp)
    sims,tsim,nsim,unique=molgen.evaluate_mols_similarity_to_given_fps(mols,target_fps)
    lstr=f'| temp: {temp} tsmi: {tsim:.3E} nsim: {nsim:.3E} unknown: {unique:.3E} validity: {validity:.3E}'
    print (lstr)
    Save_smiles_list(smis,f'{picpath}/ft_temp_{temp:.2E}.smi')
    img=Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(250,250),legends=list([f'{float(a):.3E}' for a in sims]))
    img.save(f'{picpath}/ft_temp_{temp:.2E}.png')

