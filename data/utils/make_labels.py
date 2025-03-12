"""
make_labels.py: Compute and extract labels 
Omid Mokhtari - Inria 2025
This file is part of DynamicGT.
Released under CC BY-NC-SA 4.0 License
"""

import requests
import os
import gemmi
import tempfile
import Bio.PDB
from Bio.PDB import PDBParser, DSSP, Structure, Model


def parse_pdb_info(filepath):
    filename = filepath.split('/')[-1]
    pdb_id, chain_id_with_extension = filename.split('_')
    chain_id = chain_id_with_extension.replace('.pdb', '')
    return pdb_id, chain_id
    
def cif_to_pdb(cif_file, pdb_file):
    structure = gemmi.read_structure(cif_file)
    structure.write_pdb(pdb_file)

def fetch_pdb(pdb_id):
    # Define the PDB and CIF download URLs
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"

    # Collect all pdb files here and you should remove all "not polymer.protein" to avoid labeling not protein type entities
    base_dir = "/home/omokhtar/Desktop/final_residue/data/rcsb_tmp/"
    os.makedirs(base_dir, exist_ok=True)

    pdb_file_path = os.path.join(base_dir, f"{pdb_id}.pdb")
    cif_file_path = os.path.join(base_dir, f"{pdb_id}.cif")


    if os.path.exists(pdb_file_path):
        return pdb_file_path
    try:
        response = requests.get(pdb_url)
        response.raise_for_status()
        with open(pdb_file_path, 'wb') as f:
            f.write(response.content)
        return pdb_file_path

    # If PDB download fails, attempt to download CIF and convert it to PDB
    except requests.exceptions.RequestException as e:
        print(f"PDB file not found, attempting to download CIF for {pdb_id}...")

        try:
            response = requests.get(cif_url)
            response.raise_for_status()

            with open(cif_file_path, 'wb') as f:
                f.write(response.content)
            cif_to_pdb(cif_file_path, pdb_file_path)
            return pdb_file_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading CIF file: {e}")
            return None

def filter_chain_to_structure(pdb_file, target_chain):
    """
    Extracts a specific chain from a PDB file and returns the modified structure object.
    """
    parser = PDBParser()
    structure = parser.get_structure("Original_Structure", pdb_file)
    
    filtered_structure = Structure.Structure("Filtered")
    model = next(structure.get_models())  # Assuming the structure contains only one model
    
    # Create a new model with the same ID
    filtered_model = Model.Model(model.id)
    filtered_structure.add(filtered_model)
    
    # Adding only the target chain to the new model
    for chain in model:
        if chain.id == target_chain:
            new_chain = chain.copy()
            new_chain.detach_parent()  # Detach from the original parent
            filtered_model.add(new_chain)
    return filtered_structure
    
def split_pdb_by_chain_to_files(pdb_path):
    doc = gemmi.read_structure(pdb_path)
    chain_structures = {}

    # Initialize structures for each chain across all models
    for model in doc:
        for chain in model:
            if chain.name not in chain_structures:
                chain_structures[chain.name] = gemmi.Structure()

            new_model = gemmi.Model(f'model_{model.name}')
            new_model.add_chain(chain.clone())
            chain_structures[chain.name].add_model(new_model)

    chain_files = {}
    for chain_id, structure in chain_structures.items():
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdb', mode='w')
        structure.write_pdb(temp_file.name)
        temp_file.close()
        chain_files[chain_id] = temp_file.name
    return chain_files
   
            
            
            
            
            
            
            
            

