"""
extract_loss_dist.py: Calculating distances for loss function
Omid Mokhtari - Inria 2025
This file is part of DynamicGT.
Released under CC BY-NC-SA 4.0 License
"""

import h5py
import pickle
from loss_dist import get_dist
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import time


def parse_fasta(file_path):
    sequences = {}
    current_id = None
    current_seq = []

    with open(file_path, 'r') as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:] 
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    return sequences

def each_process(process_args):
	each, labels, seqs = process_args
	start_time = time.time()  # Start the timer
	try:
		pdb_id, chain_id = each.split('_')
		seq = seqs.get(each, 'ID not found')
		if len(labels)>400:
			return (each, None, 'TOO BIG!')
		assert len(seq)==len(labels), f"{len(seq)}\n{len(labels)}"
		dists = get_dist(pdb_id, chain_id, labels, seq)
		return (each, dists, None)
	except Exception as e:
		print(f"Error processing {pdb_id}: {e}")
		return (each, None, str(e))
	finally:
		elapsed_time = time.time() - start_time
		if elapsed_time > 40:
			print(f"Time taken for {each} size {len(seq)}:  {elapsed_time:.4f} seconds")  # Print the time taken

def parallelize_processing(list_of_ids, list_of_labels, seqs, n_processes=None):
    if n_processes is None:
        n_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave **one** core free
        
    # arguments for each process
    process_args = [(key, list_of_labels[key], seqs) for key in list_of_labels]
    
    dist_dict = {}
    not_done = []
    with tqdm(total=len(process_args), desc="Processing Items") as pbar:
        with Pool(processes=n_processes) as pool:
            for each, dist, error in pool.imap_unordered(each_process, process_args):
                if error:
                    not_done.append(f"{each}\t{error}\n")
                else:
                    dist_dict[each] = dist
                pbar.update(1)
    return dist_dict, not_done		

if __name__ == '__main__':
	r = h5py.File("/home/omokhtar/Desktop/PPI/data/db_benchmarks_with_dists.h5",'r')
	fasta_path = '/home/omokhtar/Desktop/PPI/data/splitting_af/benchmarks.txt'
	seqs = parse_fasta(fasta_path)
	data=r['data']['labels']
	list_of_ids = list(data.keys())
	list_of_labels = {each: list(data[each]['label']) for each in list_of_ids}

	dist_dict, not_done = parallelize_processing(list_of_ids, list_of_labels, seqs, n_processes=None)
	print(f"Successfully processed: {len(dist_dict)} items")
	print(f"Failed to process: {len(not_done)} items")
		
	
	with open('/home/omokhtar/Desktop/PPI/data/dists_benchmarks.pkl', 'wb') as pkl_file:
		pickle.dump(dist_dict, pkl_file)
	with open('/home/omokhtar/Desktop/PPI/data/not_done_dists.txt', 'w') as tsv_file:
		tsv_file.write(''.join([i for i in not_done]))

