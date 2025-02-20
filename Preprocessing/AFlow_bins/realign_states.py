from Bio.PDB import PDBParser, PDBIO, Superimposer
from sys import argv
import os

def realign_nmr_pdb(input_pdb_file, output_pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('NMR_Structure', input_pdb_file)

    models = list(structure.get_models())
    reference_model = models[0]  # Choose the first model as reference

    # Use CA atoms for superimposition
    ref_atoms = [atom for atom in reference_model.get_atoms() if atom.id == 'CA']

    super_imposer = Superimposer()

    for model in models[1:]:  # Skip the first model since it's the reference
        alt_atoms = [atom for atom in model.get_atoms() if atom.id == 'CA']

        # Ensure the number of atoms matches in both models
        if len(ref_atoms) == len(alt_atoms):
            super_imposer.set_atoms(ref_atoms, alt_atoms)
            super_imposer.apply(model.get_atoms())

    # Writing the aligned models to a new PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_file)

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdb'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{filename}")
            
            try:
                realign_nmr_pdb(input_path, output_path)
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")



if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python script_name.py input_pdb_file")
    else:
        input_pdb_file = argv[1]
        output_pdb_file = input_pdb_file + 'realigned'
        if not os.path.exists(output_pdb_file):
            os.makedirs(output_pdb_file)
        process_directory(input_pdb_file, output_pdb_file)
        print(f"Output saved as: {output_pdb_file}")
