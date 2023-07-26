import torch
from FBG_3DGen.model import * 
from FBG_3DGen.comparm import * 
import argparse as arg
import pickle,os
from tqdm import tqdm 
parser=arg.ArgumentParser(description='Graph-CRealNVP')
parser.add_argument('-i','--input')
parser.add_argument('--gpu')
args=parser.parse_args()
jsonfile=args.input
gpu=args.gpu
UpdateGPARAMS(jsonfile)
os.environ["CUDA_VISIBLE_DEVICES"] =gpu
"""
train_MFTs=[]
for i in range(10):
    with open(GP.trainsetting.dataset_path+f'/MFTs_saved_{GP.syssetting.ring_cover_rate}_{i}.pickle','rb') as f:
        MFTs=pickle.load(f)
        print (len(MFTs))
        MFTs60=[]
        for mft in MFTs:    
            if mft.natoms<60:
                MFTs60.append(mft)
    with open(GP.trainsetting.dataset_path+f'/MFTs_saved_{GP.syssetting.ring_cover_rate}_{i}_less60.pickle','wb') as f:
        pickle.dump(MFTs60,f)
    train_MFTs+=MFTs60

#cutnum=math.ceil(GP.trainsetting.cut*len(train_MFTs))
#Statistic_dataset_params(train_MFTs)
MFTsfilelist=[f"./datasets/MFTs_saved_0.97_{i}_less60.pickle" for i in range(10)]
pid=0
for fname in tqdm(MFTsfilelist):
    with open(fname,'rb') as f:
        MFTs=pickle.load(f)
    nparts=math.ceil(len(MFTs)/25000)
    for i in range(nparts):
        with open(f'./datasets/MFTs_chembl_{pid}.pickle','wb') as f:
            pickle.dump(MFTs[i*25000:(i+1)*25000],f)
            pid+=1
New_MFTsfilelist=[f'./datasets/MFTs_chembl_{i}.pickle' for i in range(pid)]
"""
datapath=
New_MFTsflist=[f'./datasets/MFTs_Train_saved_{i}_dealed.pickle' for i in range(51)]+\
    [f'./datasets/MFTs_Valid_saved_{i}_dealed.pickle' for i in range(4)]
    #+[f'./datasets/MFTs_Test_saved_{i}_dealed.pickle' for i in range(10)]
molgen=MolGen_Model(modelname="MFTs_Gen_Model",loadtype="Perepoch")
molgen.fit_for_big_datasets(New_MFTsflist,epochnum=5000,init_epoch=16)

