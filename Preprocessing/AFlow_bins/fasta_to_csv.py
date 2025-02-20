from datetime import datetime
import csv

def parse_fasta_to_csv(input_file, output_file):
    records = []
    current_name = ""
    current_seq = []
    
    # Read FASTA file
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name and current_seq:
                    records.append({
                        'name': current_name,
                        'seqres': ''.join(current_seq),
                        'release_date': datetime.now().strftime('%Y-%m-%d'), 
                        'msa_id': current_name, 
                        'seqlen': len(''.join(current_seq))
                    })
                current_name = line[1:]  # remove '>' character
                current_seq = []
            else:
                current_seq.append(line)
        
        # Don't forget to add the last sequence
        if current_name and current_seq:
            records.append({
                'name': current_name,
                'seqres': ''.join(current_seq),
                'release_date': datetime.now().strftime('%Y-%m-%d'),  # placeholder date
                'msa_id': current_name,  # same as name as specified
                'seqlen': len(''.join(current_seq))
            })
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['name', 'seqres', 'release_date', 'msa_id', 'seqlen']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(records)

if __name__ == "__main__":
    parse_fasta_to_csv("splits/luiz.fa", "splits/luiz.csv")
