"""
pdb_format.py: Helper function to process PDB files 
Omid Mokhtari - Inria 2025
This file is part of DynamicGT.
Released under CC BY-NC-SA 4.0 License
"""

import os
import re

def process_pdb_file(input_file, output_file):
    ### Process a PDB file by standardizing MODEL line spacing.
    remove_keywords = ['TITLE', 'REMARK', 'CRYST1', 'TER']
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    processed_lines = []
    for line in lines:
        # Skip lines that start with keywords to be removed
        if any(line.startswith(keyword) for keyword in remove_keywords):
            continue
            
        # Standardize MODEL lines
        if line.startswith('MODEL'):
            model_num = re.search(r'MODEL\s*(\d+)', line)
            if model_num:
                # 7 spaces between MODEL and number
                processed_line = f"MODEL{' ' * 6}{model_num.group(1)}\n"
                processed_lines.append(processed_line)
            continue
            
        processed_lines.append(line)
    
    with open(output_file, 'w') as f:
        f.writelines(processed_lines)

def process_directory(input_dir, output_dir):
    """
    Process all PDB files in the input directory and save them to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdb'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{filename}")
            
            try:
                process_pdb_file(input_path, output_path)
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    input_directory = "/home/omokhtar/Desktop/final_atom/data/benchmarks/final_ag/representatives"
    output_directory = "/home/omokhtar/Desktop/final_atom/data/benchmarks/final_ag/representatives2"
    process_directory(input_directory, output_directory)
