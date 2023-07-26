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

#Statistic_dataset_params(MFTs)
molgen=MolGen_Model(modelname='MFTs_Gen_Model',loadtype="Perepoch")
total_mols=[]
total_smis=[]
os.system(f'mkdir -p ./sampled_mols')
with open('sampled_mols.smi','w') as f:
    for i in tqdm(range(1)):
        #try:
            molnodes,moledges,molatomnums,mol_likelihoods=molgen.sample_batch()
            mols,smis=molgen.nodes_edges_to_mol(molnodes,moledges,molatomnums)
            total_mols+=mols
            total_smis+=smis
            img=Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(250,250))
            img.save(f'./sampled_mols/sampled_mols_{i}.png')
            for smi in smis:
                f.write(f'{smi}\n')
        #except:
        #    print (f'{i} failed!!')
"""
molwts,qeds,tpsas,logps,hbas,hbds=analysis_molecules_properties(total_smis)
FcMol_max_ring_size,FcMol_max_single_ring_size,frag_type_statistics,frag_smis_statistics=analysis_molecules_ringsystems(total_smis)
guacamol_smis=Load_smiles_list('guacamol_v1_train.smiles')
guacamol_smis=random.sample(guacamol_smis,50*200)
guacamol_max_ring_size,guacamol_max_single_ring_size,frag_type_statistics,frag_smis_statistics=analysis_molecules_ringsystems(guacamol_smis)
datadict={"FcMol":{"data":{"FcMol":FcMol_max_ring_size,},"xlabel":"Max ring size in molecules","ylabel":"Distribution","style":"distplot","xlim":(0,30),"ylim":(0,0.5)},
            "GuacaMol":{"data":{'GuacaMol Datasets':guacamol_max_ring_size},"xlabel":"Max ring size in molecules","ylabel":"Distribution","style":"distplot","xlim":(0,30),"ylim":(0,0.5)}}

Define_box_grid_figure(datadict,grids=(2,1),filename='./FcMol_with_ringsize_control2.png')
datadict={"FcMol":{"data":{"FcMol":FcMol_max_single_ring_size,},"xlabel":"Max single ring size in molecules","ylabel":"Distribution","style":"distplot","xlim":(0,30),"ylim":(0,0.5)},
            "GuacaMol":{"data":{'GuacaMol Datasets':guacamol_max_single_ring_size},"xlabel":"Max single ring size in molecules","ylabel":"Distribution","style":"distplot","xlim":(0,30),"ylim":(0,0.5)}}
Define_box_grid_figure(datadict,grids=(2,1),filename='./FcMol_single_ring_size_withcontrol_2.png')
"""
