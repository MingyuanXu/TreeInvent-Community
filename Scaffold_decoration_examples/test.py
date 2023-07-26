import rdkit 
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle 

with open('scaffold.smi','r') as f:
    smis=[line.strip() for line in f.readlines()]
    for smi in smis:
        mol=Chem.MolFromSmiles(smi)
        with open('scaffold.pickle','wb') as f:
            pickle.dump(mol,f)
        Chem.AllChem.Compute2DCoords(mol)
        #Chem.rdDepictor.Compute2DCoords(mol)
        Chem.MolToPDBFile(mol,'scaffold.pdb')


