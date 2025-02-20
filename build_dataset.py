import numpy as np
import torch as pt
from tqdm import tqdm
import h5py
import warnings 
import requests
import argparse
from pathlib import Path
from typing import List, Dict, Any
from glob import glob
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from utils.PDB_processing import StructuresDataset, std_elements
from utils.feature_extraction import encode_sequence


warnings.simplefilter('ignore', PDBConstructionWarning)
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process protein structure data')
    parser.add_argument('-i', '--input', 
                        required=True,
                        help='Input directory containing PDB files (*.pdb)')
    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--num-workers',
                        type=int,
                        default=0,
                        help='Number of worker processes for data loading')
    parser.add_argument('--min-sequence-length',
                        type=int,
                        default=64,
                        help='Minimum sequence length to process')
    return parser.parse_args()
    
def process_structures(input_path: str, output_path: str, 
                      num_workers: int,
                      min_sequence_length: int) -> None:
    """Process protein structures and save to HDF5 database."""
    # Get all PDB files in the input directory
    structure_files = glob(str(Path(input_path) / "*.pdb"))
    if not structure_files:
        raise ValueError(f"No PDB files found in {input_path}")

    # Initialize dataset and dataloader
    dataset = StructuresDataset(structure_files)
    dataloader = pt.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers
    )

    device = pt.device("cuda")
    metadata = []

    with h5py.File(output_path, 'w', libver='latest') as db:
        pbar = tqdm(dataloader, desc="Processing structures")
        
        for features_dic, sasa_dic_unbound, labels_dic, pdb_file in pbar:
            for each_chain, features in features_dic.items():
                # Parse the IDs
                pdbID, chainID = (pdb_file.split('.')[0].split('_') 
                                if '_' in pdb_file 
                                else (pdb_file.split('.')[0], each_chain))
                
                if f"data/features/{pdbID}_{chainID}" in db:
                    continue

                # Extract Features
                (aa_map, seq, atom_type, D_nn, R_nn, motion_v_nn, 
                 motion_s_nn, rmsf1, rmsf2, CP_nn, nn_topk) = features
                onehot_seq = pt.tensor(encode_sequence(atom_type, std_elements))
                rsa = sasa_dic_unbound[chainID]
                label = labels_dic[each_chain]

                # Validate dimensions
                try:
                    assert (onehot_seq.shape[0] == CP_nn.shape[0] == 
                           rsa.shape[0] == np.array(aa_map).shape[0]), \
                           (f"Dimension mismatch in {pdb_file}: "
                            f"{onehot_seq.shape[0]}, {rsa.shape[0]}, "
                            f"{np.array(aa_map).shape[0]}")
                    assert aa_map[-1] == len(seq) == len(label), \
                           (f"Length mismatch in {pdb_file}: "
                            f"{aa_map[-1]}, {len(seq)}, {len(label)}")
                except AssertionError as e:
                    print(f"Skipping {pdb_file} due to: {str(e)}")
                    continue

                # Filter small sequences and those without interface
                if (not np.any(np.array(label) == 1) or 
                    onehot_seq.shape[0] < min_sequence_length):
                    continue

                # Save sequence to FASTA file
                with open("seqs.fasta", 'a') as fasta_file:
                    fasta_file.write(f">{pdbID}_{chainID}\n{seq}\n")

                # Store features and labels
                metadata.append({
                    'ID': f"{pdbID}_{chainID}",
                    'size': (len(aa_map), aa_map[-1]),
                    'seq': seq
                })

                # Create feature and label groups
                feature_group = db.create_group(f"data/features/{pdbID}_{chainID}")
                label_group = db.create_group(f"data/labels/{pdbID}_{chainID}")

                # Save features
                feature_group.create_dataset("aa_map", data=aa_map)
                feature_group.create_dataset("onehot_seq", data=onehot_seq)
                feature_group.create_dataset("rmsf1", data=rmsf1.unsqueeze(1))
                feature_group.create_dataset("rmsf2", data=rmsf2.unsqueeze(1))
                feature_group.create_dataset("CP_nn", 
                    data=CP_nn.to(dtype=pt.float32).unsqueeze(2))
                feature_group.create_dataset("rsa", data=rsa)
                feature_group.create_dataset("nn_topk", 
                    data=nn_topk.to(dtype=pt.int64))
                feature_group.create_dataset("D_nn", 
                    data=D_nn.to(dtype=pt.float32).unsqueeze(2))
                feature_group.create_dataset("R_nn", 
                    data=R_nn.to(dtype=pt.float32))
                feature_group.create_dataset("motion_v_nn", 
                    data=motion_v_nn.to(dtype=pt.float32))
                feature_group.create_dataset("motion_s_nn", 
                    data=motion_s_nn.to(dtype=pt.float32).unsqueeze(2))

                # Save label
                label_group.create_dataset("label", 
                    data=label.to(pt.float32).unsqueeze(1))

        # Save metadata
        db['metadata/ID'] = np.array([m['ID'] for m in metadata]).astype(np.string_)
        db['metadata/size'] = np.array([m['size'] for m in metadata])
        db['metadata/seq'] = np.array([m['seq'] for m in metadata], dtype='S')

def main():
    args = parse_arguments()
    
    try:
        process_structures(
            input_path=args.input,
            output_path=args.output,
            num_workers=args.num_workers,
            min_sequence_length=args.min_sequence_length
        )
        print(f"Successfully processed structures. Output saved to {args.output}")
    except Exception as e:
        print(f"Error processing structures: {str(e)}")
        raise

if __name__ == "__main__":
    main()
