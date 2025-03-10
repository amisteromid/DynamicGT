from Bio import SeqIO, PDB
import os
from Bio.PDB import *
import numpy as np



def p_to_bfactor(dic, device, save_path):
    for chain_key, values in dic.items():
        print(f"\nProcessing {chain_key}")
        pdb_path = chain_key
        p_values = values[0]
        input_seq = values[1]
        
        # Load the structure
        parser = PDB.PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(chain_key, pdb_path)
        except FileNotFoundError:
            print(f"Error: PDB file {pdb_path} not found")
            continue
            
        # Extract sequence from first model only
        first_model = next(structure.get_models())
        extracted_seq = ""
        for chain in first_model:
            for residue in chain:
                if 'CA' in residue:
                    resname = residue.get_resname()
                    # Convert three-letter code to one-letter code
                    one_letter = PDB.Polypeptide.three_to_one(resname)
                    extracted_seq += one_letter
        
        # Verify sequence match
        if extracted_seq != input_seq:
            print(f"Warning: Sequence mismatch for {chain_key}")
            print(f"Extracted: {extracted_seq}")
            print(f"Provided: {input_seq}")
            continue
        
        # Update B-factors for all models
        for model in structure:
            residue_index = 0
            for chain in model:
                for residue in chain:
                    if residue_index >= len(p_values):
                        break
                    # Set B-factor for all atoms in the residue
                    for atom in residue:
                        atom.set_bfactor(float(p_values[residue_index]))
                    residue_index += 1
        
        # Save modified structure (includes all models)
        os.makedirs(save_path, exist_ok=True)
        output_path = os.path.join(save_path, f"{chain_key}_modified.pdb")
        io = PDB.PDBIO()
        io.set_structure(structure)
        try:
            io.save(output_path)
            print(f"Saved modified structure to {output_path}")
        except Exception as e:
            print(f"Error saving file {output_path}: {str(e)}")
	
	
	
	
	
