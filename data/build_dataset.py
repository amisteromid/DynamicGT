import numpy as np
import torch as pt
from tqdm import tqdm
import h5py
import warnings 
import requests

from utils.PDB_processing import StructuresDataset
from utils.configs import dataset_configs
from utils.feature_extraction import encode_sequence
from utils.protein_chemistry import list_aa, std_elements

#from Bio.PDB.PDBExceptions import PDBConstructionWarning
#warnings.simplefilter('ignore', PDBConstructionWarning)
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
	dataset = StructuresDataset(dataset_configs['structures_folder'])
	dataloader = pt.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, pin_memory=False, num_workers=4, prefetch_factor=2) #num_workers=8, prefetch_factor=4
	
	device = pt.device("cuda")
	
	with h5py.File(dataset_configs['dataset_filepath'], 'w', libver='latest') as db:
		metadata = []
		
		pbar = tqdm(dataloader)
		count= 0
		for features_dic, sasa_dic_unbound, labels_dic, pdb_file in pbar:
			for each_chain,features in features_dic.items():	
				# Parse the IDs
				pdbID, chainID = pdb_file.split('.')[0].split('_') if '_' in pdb_file else (pdb_file.split('.')[0], each_chain)
				if f"data/features/{pdbID}_{chainID}" in db: continue
				# Extract Features
				aa_map, seq, atom_type, D_nn, R_nn, motion_v_nn, motion_s_nn, rmsf1, rmsf2, CP_nn, nn_topk = features
				onehot_seq = pt.tensor(encode_sequence(atom_type, std_elements))
				rsa = sasa_dic_unbound[chainID]
				label = labels_dic[each_chain]
				
				# Check Check Check
				assert onehot_seq.shape[0]==CP_nn.shape[0]==rsa.shape[0]==np.array(aa_map).shape[0],f"{pdb_file},{onehot_seq.shape[0]},{rsa.shape[0]},{np.array(aa_map).shape[0]}"
				assert aa_map[-1] == len(seq)==len(label), f"{aa_map[-1]},{len(seq)},{len(label)}"
				
				# Filter small ones and those without interface
				if not np.any(np.array(label) == 1) or onehot_seq.shape[0] < 64: continue
				with open("seqs.fasta", 'a') as fasta_file: fasta_file.write(f">{pdbID}_{chainID}\n{seq}\n")
				
				# Store them
				metadata.append({'ID': f"{pdb_id}_{chain_id}",'size': (len(aa_map), aa_map[-1]),'seq': seq})	
				feature_group = db.create_group(f"data/features/{pdbID}_{chainID}")
				label_group = db.create_group(f"data/labels/{pdbID}_{chainID}")
				
				feature_group.create_dataset("aa_map", data=aa_map)
				feature_group.create_dataset("onehot_seq", data=onehot_seq)
				feature_group.create_dataset("rmsf1", data=rmsf1.unsqueeze(1))
				feature_group.create_dataset("rmsf2", data=rmsf2.unsqueeze(1))
				feature_group.create_dataset("CP_nn", data=CP_nn.to(dtype=pt.float32).unsqueeze(2))
				feature_group.create_dataset("rsa", data=rsa)
				feature_group.create_dataset("nn_topk", data=nn_topk.to(dtype=pt.int64))
				feature_group.create_dataset("D_nn", data=D_nn.to(dtype=pt.float32).unsqueeze(2))
				feature_group.create_dataset("R_nn", data=R_nn.to(dtype=pt.float32))
				feature_group.create_dataset("motion_v_nn", data=motion_v_nn.to(dtype=pt.float32))
				feature_group.create_dataset("motion_s_nn", data=motion_s_nn.to(dtype=pt.float32).unsqueeze(2))
				label_group.create_dataset("label", data=label.to(pt.float32).unsqueeze(1))
					
		db['metadata/ID'] = np.array([m['ID'] for m in metadata]).astype(np.string_) #6wa1_A
		db['metadata/size'] = np.array([m['size'] for m in metadata]) #num_res
		db['metadata/seq'] = np.array([m['seq'] for m in metadata], dtype='S') #seq
