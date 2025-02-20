import sys

def convert_to_fasta(input_file):
    output_file = input_file.rsplit(".", 1)[0] + ".fasta"
    
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        # Skip the header line
        next(infile)
        for line in infile:
            name, seqres, _, _, _ = line.strip().split(",")
            outfile.write(f">{name}\n{seqres}\n")
    
    print(f"FASTA file has been written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 script.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    convert_to_fasta(input_file)

