import os
from pymol import cmd

cmd.reinitialize()

pdb_dir = "modified_training_extended1"
output_fasta = "modified_training_extended1.fasta"

with open(output_fasta, "w") as fasta_file:
    for filename in os.listdir(pdb_dir):
        if filename.endswith(".pdb"):
            # Extract pdbid_chainid from filename
            pdb_chain = filename.replace(".pdb", "")
            
            cmd.load(os.path.join(pdb_dir, filename))
            
            # Using 'polymer.protein' to ensure we only get protein sequence
            seq = cmd.get_fastastr('polymer.protein')
            
            # Format sequence header and write to file
            seq_clean = seq.split('\n', 1)[1].replace('\n', '')
            fasta_file.write(f">{pdb_chain}\n{seq_clean}\n")
            
            cmd.delete('all')

print(f"Sequences have been written to {output_fasta}")
