import os
from sys import argv

def process_pdb_file(input_file, output_file):
    """
    Process a PDB file by removing specific lines and standardizing MODEL line spacing.
    """
    remove_keywords = ['TITLE', 'REMARK', 'CRYST1', 'TER']
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    processed_lines = []
    model_count = 1  # Initialize counter for MODEL lines
    for line in lines:
        # Skip lines that start with keywords to be removed
        if any(line.startswith(keyword) for keyword in remove_keywords):
            continue
            
        # Standardize MODEL lines
        if line.startswith('MODEL'):
            processed_line = f"MODEL{' ' * 6}{model_count}\n"
            processed_lines.append(processed_line)
            model_count += 1
            continue
        
        processed_lines.append(line)
    
    with open(output_file, 'w') as f:
        f.writelines(processed_lines)

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdb'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{filename[:4].upper() + filename[4:]}")
            
            try:
                process_pdb_file(input_path, output_path)
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python3 script_name.py input_pdb_file")
    else:
        input_directory = argv[1]
        output_directory = input_directory + 'formatted'
        print (output_directory)
        process_directory(input_directory, output_directory)
