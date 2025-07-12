import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#base_path = '/home/omokhtar/Desktop/PPI/data/splitting_af/'

# Load the data
cluster_df = pd.read_csv('clustered_0.3.tsv', sep='\t', header=None, names=['representative', 'individual'])
'''
benchmarksZDOCK = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_zdock.txt', dtype=np.dtype('U')))))
benchmarks53 = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_53.txt', dtype=np.dtype('U')))))
benchmarksAG = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_ag.txt', dtype=np.dtype('U')))))
benchmarks60 = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_60.txt', dtype=np.dtype('U')))))
benchmarksMFIB = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_MFIB1.txt', dtype=np.dtype('U')))))
benchmarksIDR = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_idrbind.txt', dtype=np.dtype('U')))))
benchmarksFuzDB = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_fuzdb1.txt', dtype=np.dtype('U')))))
benchmarksBret42 = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_bret42.txt', dtype=np.dtype('U')))))
benchmarks = np.concatenate([benchmarksZDOCK, benchmarks53, benchmarksAG, benchmarks60, benchmarksMFIB, benchmarksFuzDB, benchmarksIDR, benchmarksBret42])
'''

# Create clusters from tsv
clusters = {}
for _, row in cluster_df.iterrows():
    rep = row['representative']
    ind = row['individual']
    if rep not in clusters:
        clusters[rep] = set()
    clusters[rep].update([rep, ind]) 
    
'''
train_clusters=[]
bm_clusters=[]
bm_entries=[]
for _,listt in clusters.items():
    if any(item in listt for item in benchmarks):
        bm_clusters.append(listt)
        bm_entries.extend(listt)
    else:
        train_clusters.append(listt)
'''   

# Split the clusters into training, validation, and test sets
train_clusters, valid_clusters = train_test_split(list(clusters.keys()), test_size=0.1, random_state=67)
#train_clusters, val_clusters = train_test_split(list(clusters.keys()), test_size=0.01, random_state=42) #3--> 112 1-->42 4-->53 2-->67

train_entries = [item for rep in train_clusters for item in clusters[rep]]
valid_entries = [item for rep in valid_clusters for item in clusters[rep]]

print (f"Training set: {len(train_clusters)} clusters --> {len(train_entries)} entries")
print (f"Validation set: {len(valid_clusters)} clusters --> {len(valid_entries)} entries")
#print (f"Benchmarks set: {len(bm_clusters)} clusters --> {len(bm_entries)} entries")


# Save the results to files
with open('train.txt', 'w') as f:
    for item in train_entries:
        f.write(f"{item}\n")

with open('valid.txt', 'w') as f:
    for item in valid_entries:
        f.write(f"{item}\n")

