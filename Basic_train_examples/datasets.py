from FBG_3DGen.graphs import *
from FBG_3DGen.model import *
from tqdm import tqdm 
import os
import pickle,os

def MolsToMFTs(mols,fpath='MFTs.pickle',jid=0):
    MFTs=[]
    for mol in tqdm(mols):
        if mol:
            try:
                smi=Chem.MolToSmiles(mol)
                MFTree=MolFragTree(mol,smi=smi)
                max_ringnum=0
                for fid,frag in enumerate(MFTree.clique_mols):
                    frag_natoms=len(MFTree.clique_inner_f_atoms[fid])
                    if frag_natoms>1:
                        f=MFTree.f_cliques[fid]
                        ringnum=f[0]
                        if ringnum>max_ringnum:
                            max_ringnum=ringnum
                if max_ringnum<5:
                    MFTs.append(MFTree)
            except Exception as e:
                print (f'{smi} trans to MFT failed due to {e}!')
    with open(fpath,'wb')  as f:
        pickle.dump(MFTs,f)
    print (f'{jid} process have trans {len(MFTs)} mols and saved in {fpath}')
    return

def SmisToMFTs(smis,picklepath='MFTs.pickle',savepath=f'./datasets/mol.smi',jid=0):
    mols=prepare_mols_from_smi(smis,savepath)
    MolsToMFTs(mols,picklepath,jid)
    return 

def MolsToMFTs_multi(mols,picklepath='./datasets',nproc=28,nmols_per_proc=50000):
    from multiprocessing import Pool,Queue,Manager,Process
    manager=Manager()
    DQueue=manager.Queue()
    p=Pool(nproc)
    nmols=len(mols)
    njobs=math.ceil(nmols/nmols_per_proc)
    resultlist=[]
    for i in range(njobs):
        result=p.apply_async(MolsToMFTs,(mols[i*nmols_per_proc:(i+1)*nmols_per_proc],path+f'_{i}.pickle',i))
        resultlist.append(result)
    for i in range(len(resultlist)):
        tmp=resultlist[i].get()
        print (tmp)
    p.terminate()
    p.join()
    print (f'Mols have all trans to MFTs in {path}')
    return

def SmisToMFTs_multi(smis,picklepath='./datasets/MFTs',smisavepath='./datasets/Mol',nproc=28,nmols_per_proc=25000):
    from multiprocessing import Pool,Queue,Manager,Process
    manager=Manager()
    DQueue=manager.Queue()
    p=Pool(nproc)
    nmols=len(smis)
    njobs=math.ceil(nmols/nmols_per_proc)
    resultlist=[]
    for i in range(njobs):
        result=p.apply_async(SmisToMFTs,(smis[i*nmols_per_proc:(i+1)*nmols_per_proc],picklepath+f'_{i}.pickle',f'{smisavepath}_{i}.smi',i))
        resultlist.append(result)
    for i in range(len(resultlist)):
        tmp=resultlist[i].get()
        print (tmp)
    p.terminate()
    p.join()
    print (f'SMILEs have all trans to MFTs in {path}')
    return 

path='guacamol_v1_train.smiles'
trainsmis=Load_smiles_list(path)
#trainmols=prepare_mols_from_smi(path,savepath='./datasets/train.smi')
SmisToMFTs_multi(trainsmis,picklepath='./datasets/MFTs_Train_saved',smisavepath='./datasets/Train',nproc=15)
path='guacamol_v1_valid.smiles'
validsmis=Load_smiles_list(path)
SmisToMFTs_multi(validsmis,picklepath='./datasets/MFTs_Valid_saved',smisavepath='./datasets/Valid',nproc=15)
path='guacamol_v1_test.smiles'
testsmis=Load_smiles_list(path)
SmisToMFTs_multi(testsmis,picklepath='./datasets/MFTs_Test_saved',smisavepath='./datasets/Test',nproc=15)

MFTsflist=[f'./datasets/MFTs_Train_saved_{i}.pickle' for i in range(51)]+\
    [f'./datasets/MFTs_Valid_saved_{i}.pickle' for i in range(4)]+\
    [f'./datasets/MFTs_Test_saved_{i}.pickle' for i in range(10)]

Dataset_From_MFTsflist(MFTsflist,path='./datasets',cut=0.97)
Statistic_dataset_params_from_MFTsflist(MFTsflist)




