import os
import glob
from pymol import cmd

parent_folder = '/media/omokhtar/Expansion/Omid/DynRepo/PDBbind'

# Iterate through PDB files in the specified directory
target_files = glob.glob('/home/omokhtar/Desktop/*.pdb')

for target_file in target_files:
    cmd.reinitialize()
    target = os.path.basename(target_file).split('.')[0]
    cmd.load(target_file, target)
   
    # Load another PDB file for the target
    structure_path = f'/media/omokhtar/Expansion/Omid/DynRepo/PDBbind/{target}/analysis/prot.pdb'
    cmd.load(structure_path, 'structure')
    
    # Check if structure.pdb has one chain or not
    distinguished_chains = cmd.get_chains('structure')
    
    if len(distinguished_chains) > 1:
        # If structure.pdb has multiple chains
        chain_residues = {}
        
        for chain in distinguished_chains:
            residues = cmd.get_model(f'structure and chain {chain}').atom
            for res in residues:
                chain_residues[res.resi] = chain
        
        for resi, chain in chain_residues.items():
            cmd.alter(f'{target} and resi {resi}', f'chain="{chain}"')
        print ('more than one chain: ', target)
        for each in cmd.get_chains(target):
            print (each)

        cmd.save(f'/media/omokhtar/Expansion/Omid/DynRepo/PDBbind/{target}/analysis/{target}__.pdb', target, state=0)
    
    else:
        # If structure.pdb has one chain
        cmd.fetch(f'{target[:4]}')
        which_chain = '!'
        best_rms = 9999
        
        for chain in cmd.get_chains(f'{target[:4]}'):
            rms = cmd.align(f'{target}', f'{target[:4]} and chain {chain}')[0]
            if rms < best_rms:
                which_chain = chain
                best_rms = rms
        print ('just one chain: ', target)
        for each in cmd.get_chains(target):
            print (each)
        cmd.save(f'/media/omokhtar/Expansion/Omid/DynRepo/PDBbind/{target}/analysis/{target[:4]}_{which_chain}.pdb', target, state=0)
