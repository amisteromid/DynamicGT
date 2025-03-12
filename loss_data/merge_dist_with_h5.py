"""
merge_dist_with_h5.py: Merging distance into dataset
Omid Mokhtari - Inria 2025
This file is part of DynamicGT.
Released under CC BY-NC-SA 4.0 License
"""

import h5py
import pickle
import numpy as np

h = h5py.File("/home/omokhtar/Desktop/PPI/data/db_benchmarks_v2.h5",'r')
with open('/home/omokhtar/Desktop/PPI/data/dists_benchmarks.pkl', 'rb') as file:
     p = pickle.load(file)

def fill_nan_with_adjacent_mean(data):
    arr = np.array(data)
    nan_indices = np.where(np.isnan(arr))[0]
    
    for i in nan_indices:
        left = i - 1
        right = i + 1
        
        # Find the closest non-NaN value on the left
        while left >= 0 and np.isnan(arr[left]):
            left -= 1
        
        # Find the closest non-NaN value on the right
        while right < len(arr) and np.isnan(arr[right]):
            right += 1
        
        # Calculate replacement value
        if left >= 0 and right < len(arr):
            arr[i] = (arr[left] + arr[right]) / 2
        elif left >= 0:
            arr[i] = arr[left]
        elif right < len(arr):
            arr[i] = arr[right]
    
    return arr.tolist()
    
empty = 0
with h5py.File("/home/omokhtar/Desktop/PPI/data/db_benchmarks_v2_with_dists2.h5", 'w') as new_h:
    h.copy('data', new_h)
    h.copy('metadata',new_h)
    
    for pdb_id in list(new_h['data']['labels'].keys()):
        labels = list(new_h['data']['labels'][pdb_id]['label'])
        if pdb_id in p.keys() and len(labels) == len(p[pdb_id]):
            dists = p[pdb_id]
            dists  = fill_nan_with_adjacent_mean(dists)
        else:
            empty +=1
            print(empty, pdb_id)
            dists = np.full(len(labels),0.5)
        
        group = new_h['data']['labels'][pdb_id]
        if 'dist' in group: del group['dist']
        group.create_dataset('dist', data=dists)

print("Merging complete.")
