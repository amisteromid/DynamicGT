from pymol import cmd

def process_chains():
    with open('ids_all_benchmarks.txt', 'r') as f:
        chains = [line.strip() for line in f.readlines()]
    
    with open('ids_all_benchmarks.fasta', 'w') as fasta_out:
        for chain_id in chains:
            try:
                pdb_id, chain = chain_id.split('_')
                cmd.fetch(pdb_id)
                cmd.select("chain_polymer", f"polymer and chain {chain}")
                cmd.remove("not chain_polymer")
                cmd.remove('not polymer.protein')
                seq = cmd.get_fastastr("chain_polymer")
                # Remove newlines from sequence and keep header
                header, sequence = seq.split('\n', 1)
                sequence = sequence.replace('\n', '')
                fasta_out.write(f"{header}\n{sequence}\n")
                
                cmd.delete('all')
                print(f"Processed {chain_id}")
            except Exception as e:
                print (chain_id, e)
process_chains()
print("To run: type 'process_chains' in PyMOL command line")
