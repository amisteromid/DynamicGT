"""
PDB_processing.py: Reading the input PDB file and calculating SASA 
Omid Mokhtari - Inria 2025
This file is part of DynamicGT.
Released under CC BY-NC-SA 4.0 License
"""

import os
import requests
import warnings
import tempfile

import torch as pt
import numpy as np
import Bio.PDB
from Bio.PDB import PDBParser, DSSP, Structure, Model,is_aa
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import gemmi
from Bio.Data.SCOPData import protein_letters_3to1
import json
import pandas as pd

from utils.feature_extraction import extract_dynamic_features, encode_sequence, mean_coordinates, extract_topology
from utils.make_labels import fetch_pdb,parse_pdb_info, split_pdb_by_chain_to_files, filter_chain_to_structure

std_elements = np.array([
    'C', 'O', 'N', 'S', 'P', 'Se', 'Mg', 'Cl', 'Zn', 'Fe', 'Ca', 'Na',
    'F', 'Mn', 'I', 'K', 'Br', 'Cu', 'Cd', 'Ni', 'Co', 'Sr', 'Hg', 'W',
    'As', 'B', 'Mo', 'Ba', 'Pt'
])
list_modified_aa = ['ABU', 'ACE', 'AIB', 'ALA  ', 'ARG  ', 'ARGN', 'ASN', 'ASN1  ', 'ASP', 'ASP1', 'ASPH', 'ASPP', 'ASH ', 'CT3 ', 'CYS ', 'CYS1', 'CYS2 ', 'CYSH ', 'DALA', 'GLN  ', 'GLU  ', 'GLUH ', 'GLUP', 'GLH', 'GLY', 'HIS ', 'HIS1', 'HISA ', 'HISB ', 'HISH', 'HISD', 'HISE', 'HISP ', 'HSD', 'HSE', 'HSP', 'HYP  ', 'ILE  ', 'LEU  ', 'LSN', 'LYS  ', 'LYSH ', 'MELEU', 'MET  ', 'MEVAL', 'NAC  ', 'NME ', 'NHE', 'NH2', 'PHE  ', 'PHEH ', 'PHEU ', 'PHL ', 'PRO  ', 'SER  ', 'THR  ', 'TRP  ', 'TRPH ', 'TRPU ', 'TYR  ', 'TYRH ', 'TYRU ', 'TYS ', 'VAL  ', 'PGLU', 'HID', 'HIE', 'HIP', 'LYP', 'LYN', 'CYN', 'CYM', 'CYX', 'DAB', 'ORN', 'HYP', 'NALA', 'NGLY', 'NSER', 'NTHR', 'NLEU', 'NILE', 'NVAL', 'NASN', 'NGLN', 'NARG', 'NHID', 'NHIE', 'NHIP', 'NHISD', 'NHISE', 'NHISH', 'NTRP', 'NPHE', 'NTYR', 'NGLU', 'NASP', 'NLYS', 'NORN', 'NDAB', 'NLYSN', 'NPRO', 'NHYP', 'NCYS', 'NCYS2', 'NMET', 'NASPH', 'NGLUH', 'CALA', 'CGLY', 'CSER', 'CTHR', 'CLEU', 'CILE', 'CVAL', 'CASN', 'CGLN', 'CARG', 'CHID', 'CHIE', 'CHIP', 'CHISD', 'CHISE', 'CHISH', 'CTRP', 'CPHE', 'CTYR', 'CGLU', 'CASP', 'CLYS', 'CORN', 'CDAB', 'CLYSN', 'CPRO', 'CHYP', 'CCYS', 'CCYS2', 'CMET', 'CASPH', 'CGLUH']

def split_nmr_pdb(input_pdb_file):
    structure = gemmi.read_structure(input_pdb_file)

    chains_models = {}
    # Iterate over models in the structure
    for model in structure:
        for chain in model:
            chain_type = structure[model.name][chain.name].get_polymer().check_polymer_type()
            if chain_type == gemmi.PolymerType.PeptideL or chain_type == gemmi.PolymerType.Unknown:
                chain_id = chain.name
                if chain_id not in chains_models:
                    chains_models[chain_id] = []
                chains_models[chain_id].append(chain)
    return chains_models
    
def make_pdb(chains):
    structure = gemmi.Structure()

    if isinstance(chains, gemmi.Chain):
        # Single chain, single model
        model = gemmi.Model('model_1')
        model.add_chain(chains)
        structure.add_model(model)
    elif isinstance(chains, list):
        if all(isinstance(chain, gemmi.Chain) for chain in chains):
            if all(chain.name == chains[0].name for chain in chains):
                # Same chain, multiple models
                for i, chain in enumerate(chains):
                    model = gemmi.Model(f'model_{i + 1}')
                    model.add_chain(chain)
                    structure.add_model(model)
            else:
                # Multiple chains, single model
                model = gemmi.Model('model_1')
                for chain in chains:
                    model.add_chain(chain)
                structure.add_model(model)
        else:
            raise ValueError("Invalid input: list items must be gemmi.Chain instances.")
    else:
        raise TypeError("Invalid input type. Input must be a gemmi.Chain or a list of gemmi.Chain.")

    # Generate the PDB string of the structure
    pdb_string = structure.make_pdb_string()

    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.pdb')
    temp_file.write(pdb_string)
    temp_file.flush()
    temp_file.seek(0) 

    return temp_file.name

def fill_nan_with_neighbors(sasa_array):
    series = pd.Series(sasa_array)
    series_interpolated = series.interpolate(method='linear')
    series_filled = series_interpolated.bfill().ffill()
    return series_filled.to_numpy()
    
def get_sasa_bound(pdb_filename):
    parser = PDBParser()
    structure = parser.get_structure('PDB', pdb_filename)
    model = structure[0]
    
    sasa_values = {}
    sequences = {}
    dssp = DSSP(model, pdb_filename, dssp='mkdssp')
    for key, value in dssp.property_dict.items():
        chain_id = key[0]
        sasa = value[3]
        res_name = value[1]
        if chain_id not in sasa_values:
            sasa_values[chain_id] = []
            sequences[chain_id] = ''
        if type(sasa)!=float: sasa=np.nan
        sasa_values[chain_id].append(np.round(sasa,3))
        sequences[chain_id] += res_name
            
    sasa_dic={}
    for chain, sasa_list in sasa_values.items():
        sasa_dic[chain] = np.array(sasa_list)
    return sasa_dic, sequences

def get_sasa_unbound(pdb_parent):
    chain_files = split_pdb_by_chain_to_files(pdb_parent)
    sasa_dic = {}
    sequences = {}
    
    for each_chain in chain_files:
        structure = filter_chain_to_structure(pdb_parent, each_chain)
        model = structure[0]
        try:
            dssp = DSSP(model, chain_files[each_chain], dssp='mkdssp')
            for key, value in dssp.property_dict.items():
                chain_id = key[0]
                sasa = value[3]
                res_name = value[1]
                if chain_id not in sasa_dic:
                    sasa_dic[each_chain] = []
                    sequences[each_chain] = ''
                if type(sasa)!=float: sasa=np.nan
                sasa_dic[each_chain].append(sasa)
                sequences[each_chain] += res_name
        except Exception as error:
            print (error)
            pass
            
    for chain, sasa_list in sasa_dic.items():
            sasa_dic[chain] = np.array(sasa_list)
    return sasa_dic,sequences



def read_pdb(pdb_filepath):
    parser = PDBParser()
    structure = parser.get_structure("PDB_structure", pdb_filepath)
    aa_map = []
    sequences = []
    atom_types = []
    coordinates = []

    for model in structure:
        chains = list(model.get_chains())
        assert len(chains) == 1, "More than one chain found in the model."

        chain = chains[0]
        
        aa_map_tmp = []
        seq = []
        atom_types_tmp = []
        atom_coords = []
        keys=[]
       
        for residue in chain:
            if (is_aa(residue, standard=False) or residue.resname in list_modified_aa) and residue.resname not in ['HOH', 'DOD']:
                for atom in residue:
                    key=f"{residue.id[1]}_{atom.get_name().strip()}"
                    if key not in keys and atom.element.strip() not in ['H', 'D']:
                        keys.append(key)
                        # Saving residue numbers, atom names, and atom coords
                        aa_map_tmp.append(residue.id[1])
                        seq.append(protein_letters_3to1.get(residue.resname,'X'))
                        atom_types_tmp.append(atom.element.strip())
                        atom_coords.append([atom.coord[0], atom.coord[1], atom.coord[2]])
        assert len(seq) == len(aa_map_tmp), "seq doesn't match with aa_map"
        index_set = set()
        result = []
        for char, index in zip(seq, aa_map_tmp):
            if index not in index_set:
                result.append(char)
                index_set.add(index)
        seq=result
        # reorder aa_map
        unique_values, counts = np.unique(aa_map_tmp, return_counts=True)
        new_values = np.arange(1, len(unique_values) + 1)
        aa_map_tmp_reordered = list(np.repeat(new_values, counts))
        # save for each model
        aa_map.append(aa_map_tmp_reordered)
        sequences.append(seq)
        atom_types.append(atom_types_tmp)
        coordinates.append(np.array(atom_coords, dtype=np.float32))
        
    # Check if all sequences are the same
    if all(seq == sequences[0] for seq in sequences) and all(each_atom_types == atom_types[0] for each_atom_types in atom_types) and all(each_aa_map == aa_map[0] for each_aa_map in aa_map):
        stacked_coords = np.stack(coordinates)
        return  aa_map[0], sequences[0], atom_types[0], stacked_coords
    else:
        raise ValueError("Not all sequences are the same across models.")
   

class StructuresDataset(pt.utils.data.Dataset):
    def __init__(self, structures_folder):
        super(StructuresDataset).__init__()
        self.structures_folder = structures_folder

    def __len__(self):
        return len(self.structures_folder)

    def __getitem__(self, i):
        pdb_parent = self.structures_folder[i]
        try:
            threshold = 0.04
            print(f"[INFO] Loading {pdb_parent}")
    
            pdb_chains = split_nmr_pdb(pdb_parent)
            pdb_id, chain_id_og = parse_pdb_info(pdb_parent)
            assert len(pdb_chains) == 1, f"More than one chain in {pdb_parent}"
    
            features_dic = {}
            models = list(pdb_chains.values())[0]
            pdb_file = make_pdb(models)
            aa_map, seq, atom_type, atoms_xyz = read_pdb(pdb_file)
            seq = ''.join(seq)
    
            nan_positions = np.argwhere(np.isnan(atoms_xyz))
            assert nan_positions.size == 0, f"NaNs in {pdb_parent} at {nan_positions}"
    
            mean_xyz = mean_coordinates(atoms_xyz)
            R, D = extract_topology(mean_xyz)
            knn = min(64, D.shape[0])
            D_nn, nn_topk = pt.topk(pt.tensor(D), knn, dim=1, largest=False)
            R_nn = pt.gather(pt.tensor(R), 1, nn_topk.unsqueeze(2).repeat(1, 1, R.shape[2])).to(pt.float32)
            motion_v_nn, motion_s_nn, rmsf1, rmsf2, CP_nn = extract_dynamic_features(atoms_xyz, nn_topk.numpy())
            assert D.shape[0] == len(aa_map)
    
            features = (aa_map, seq, atom_type, D_nn, R_nn, motion_v_nn, motion_s_nn, rmsf1, rmsf2, CP_nn, nn_topk)
            features_dic[chain_id_og] = features
    
            new_pdb_parent = fetch_pdb(pdb_id)
            sasa_dic_bound, labeled_seqs = get_sasa_bound(new_pdb_parent)
            sasa_dic_unbound, labeled_seqs2 = get_sasa_unbound(new_pdb_parent)
            assert labeled_seqs2[chain_id_og] == labeled_seqs[chain_id_og], f"{pdb_parent} bound/unbound mismatch"
    
            unlabeled_seqs = {key: value[1] for key, value in features_dic.items()}
            mapped_labels_dic = {}
            mapped_sasa_dic_unbound = {}
    
            for each_chain in unlabeled_seqs:
                sasa_dic_unbound[each_chain] = fill_nan_with_neighbors(sasa_dic_unbound[each_chain])
                sasa_dic_bound[each_chain] = fill_nan_with_neighbors(sasa_dic_bound[each_chain])
                assert len(labeled_seqs[each_chain]) == len(sasa_dic_unbound[each_chain])
    
                original_labels = np.array([1 if y - x >= threshold else 0 for x, y in zip(sasa_dic_bound[each_chain], sasa_dic_unbound[each_chain])])
                alignment = pairwise2.align.globalms(unlabeled_seqs[each_chain], labeled_seqs[each_chain], 2, -2, -7, -2)[0]
                unlabeled_seq_aligned, labeled_seq_aligned = alignment[0], alignment[1]
        
                mapped_labels = []
                mapped_sasa = []
                l_index = 0
                for u_residue, l_residue in zip(unlabeled_seq_aligned, labeled_seq_aligned):
                    if u_residue == '-':
                        l_index += 1
                    elif l_residue == '-':
                        mapped_labels.append(0)
                        mapped_sasa.append(mapped_sasa[-1] if mapped_sasa else sasa_dic_unbound[each_chain][l_index])
                    else:
                        mapped_labels.append(original_labels[l_index])
                        mapped_sasa.append(sasa_dic_unbound[each_chain][l_index])
                        l_index += 1
    
                mapped_sasa = fill_nan_with_neighbors(np.array(mapped_sasa))
                mapped_labels = np.array(mapped_labels)
                aa_map = np.array(features_dic[each_chain][0])
                mapped_sasa = np.array(mapped_sasa)[aa_map - 1]
    
                mapped_sasa_dic_unbound[each_chain] = mapped_sasa
                mapped_labels_dic[each_chain] = mapped_labels

            return features_dic, mapped_sasa_dic_unbound, mapped_labels_dic, pdb_parent.split('/')[-1]
    
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to process: {pdb_parent}")
            traceback.print_exc()
            #raise RuntimeError(f"Error in {pdb_parent}: {e}")
            return None, None, None, pdb_parent.split('/')[-1]
	
			
			
