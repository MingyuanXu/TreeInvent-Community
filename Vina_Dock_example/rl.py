import torch
from FBG_3DGen.model import * 
from FBG_3DGen.comparm import * 
import argparse as arg
import pickle,os
from rdkit.Chem import Draw

parser=arg.ArgumentParser(description='Graph-CRealNVP')
parser.add_argument('-i','--input')
parser.add_argument('--gpu')
args=parser.parse_args()
jsonfile=args.input
gpu=args.gpu
UpdateGPARAMS(jsonfile)
os.environ["CUDA_VISIBLE_DEVICES"] =gpu
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
score_function=ScoringFunction()
molgen_rl=MolGen_Model_RL(prior_modelname='MFTs_Gen_Model')
#molgen.fit(mfts=MFTs,epochnum=1)
molgen_rl.RL_fit(steps=1000,score_func=score_function)

