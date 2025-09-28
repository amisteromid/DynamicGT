import torch as pt
import h5py
import numpy as np
from utils.configs import config_data, config_model, config_runtime

def setup_dataloader(config_data, sids_selection_filepath):
    sids_sel = np.genfromtxt(sids_selection_filepath, dtype=np.dtype('U'))
    dataset = Dataset(config_data['dataset_filepath'])
    
    sids = np.array([key for key in dataset.ID])
    
    m = np.isin(sids, sids_sel)

    # Further filter out big ones
    sizes = np.array([s for s in dataset.size[:,1]])
    big_ones = np.where(sizes > 350)
    small_ones = np.where(sizes <30)
    m[small_ones] = False
    m[big_ones] = False

    dataset.update_mask(m)
    print ('!!!!!!!!!!!!!!!!!!!!')
    print ('dataset size: ',len(dataset))
    
    # define data loader
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=config_runtime['batch_size'], shuffle=True, num_workers=8, collate_fn=collate_batch_data, pin_memory=True, prefetch_factor=2)
    return dataloader
    
    
def collate_batch_data(batch_data):
    # collate features
    onehot_seq, rmsf1, rmsf2, rsa, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, mapping = collate_batch_features(batch_data)

    # collate targets
    dists = pt.cat([data[12] for data in batch_data])
    y = pt.cat([data[11] for data in batch_data])

    return onehot_seq, rmsf1, rmsf2, rsa, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, mapping, y, dists

def collate_batch_features(batch_data, max_num_nn=64): 
    # Pack node features
    onehot_seq, rmsf1, rmsf2, rsa = [
        pt.cat([data[i] for data in batch_data], dim=0) for i in range(4)
    ]
    mapping = pt.cat([data[10] for data in batch_data], dim=0)
    max_num_nn = min(max_num_nn, onehot_seq.shape[0])

    # Initialize tensors for nearest neighbors features
    num_res = sum(data[0].shape[0] for data in batch_data)
    nn_topk = pt.zeros((num_res, max_num_nn), dtype=pt.long, device=batch_data[0][0].device)
    D_nn = pt.zeros((num_res, max_num_nn, 1), dtype=pt.float, device=batch_data[0][0].device)
    R_nn = pt.zeros((num_res, max_num_nn, 3), dtype=pt.float, device=batch_data[0][0].device)
    motion_v_nn = pt.zeros((num_res, max_num_nn, 3), dtype=pt.float, device=batch_data[0][0].device)
    motion_s_nn = pt.zeros((num_res, max_num_nn, 1), dtype=pt.float, device=batch_data[0][0].device)
    CP_nn = pt.zeros((num_res, max_num_nn, 1), dtype=pt.float, device=batch_data[0][0].device)
    
    # Cumulative size tensor
    sizes = pt.tensor([data[0].shape[0] for data in batch_data], dtype=pt.long)
    cumsum_sizes = pt.cumsum(sizes, dim=0)
    
    for i, (size, data) in enumerate(zip(cumsum_sizes, batch_data)):
        ix1 = size
        ix0 = ix1 - data[0].shape[0]
        
        # Store nearest neighbors features
        nn_topk[ix0:ix1, :data[4].shape[1]] = data[4]
        D_nn[ix0:ix1, :data[5].shape[1], :] = data[5]
        R_nn[ix0:ix1, :data[6].shape[1], :] = data[6]
        motion_v_nn[ix0:ix1, :data[7].shape[1], :] = data[7]
        motion_s_nn[ix0:ix1, :data[8].shape[1], :] = data[8]
        CP_nn[ix0:ix1, :data[9].shape[1], :] = data[9]

    return onehot_seq, rmsf1, rmsf2, rsa, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, mapping



class Dataset(pt.utils.data.Dataset):
    def __init__(self, dataset_filepath):
        super(Dataset, self).__init__()
        # store dataset filepath
        self.dataset_filepath = dataset_filepath

        # preload data
        with h5py.File(dataset_filepath, 'r') as hf:
            # load ID, sizes and seq
            self.ID = np.array(hf["metadata/ID"]).astype(np.dtype('U'))
            self.size = np.array(hf["metadata/size"])
            self.seq = np.array(hf["metadata/seq"]).astype(np.dtype('U'))

        # set default selection mask
        self.m = np.ones(len(self.ID), dtype=bool)    
    
    def update_mask(self, m):
        # update mask
        self.m &= m # boolean vector with the size of whole dataset

    def get_largest(self):
        i = np.argmax(self.size[:,0] * self.m.astype(int))
        k = np.where(np.where(self.m)[0] == i)[0][0]
        return self[k]
        
    def __len__(self):
        return np.sum(self.m)

    def __getitem__(self, k): # k is the indice of entry
        # Adjust index k to the masked index
        masked_indices = np.where(self.m)[0] # chosen indices
        key_index = masked_indices[k]
        #print (len(masked_indices), masked_indices)
        key = self.ID[key_index]

        # load data
        with h5py.File(self.dataset_filepath, 'r') as hf:
            # hdf5 group
            hgrp_f = hf[f'data/features/{key}']
            hgrp_l = hf[f'data/labels/{key}']
            
            # atom to residue mapping
            mapping = pt.from_numpy(np.array(hgrp_f['aa_map']))

            # node features
            onehot_seq = pt.from_numpy(np.array(hgrp_f['onehot_seq']))
            rmsf1 = pt.from_numpy(np.array(hgrp_f['rmsf1']))
            rmsf2 = pt.from_numpy(np.array(hgrp_f['rmsf2']))
            rsa = pt.from_numpy(np.array(hgrp_f['rsa']))
            
            # neighbor indices
            nn_topk = pt.from_numpy(np.array(hgrp_f['nn_topk']))
            
            # edge features
            D_nn = pt.from_numpy(np.array(hgrp_f['D_nn']))
            R_nn = pt.from_numpy(np.array(hgrp_f['R_nn']))
            motion_v_nn = pt.from_numpy(np.array(hgrp_f['motion_v_nn']))
            motion_s_nn = pt.from_numpy(np.array(hgrp_f['motion_s_nn']))
            CP_nn = pt.from_numpy(np.array(hgrp_f['CP_nn']))

            # interface labels
            dists = pt.from_numpy(np.array(hgrp_l['dist']))
            y = pt.from_numpy(np.array(hgrp_l['label']))

        return onehot_seq, rmsf1, rmsf2, rsa, nn_topk, D_nn, R_nn, motion_v_nn, motion_s_nn, CP_nn, mapping, y, dists
