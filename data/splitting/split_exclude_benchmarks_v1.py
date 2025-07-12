import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

base_path = '/home/omokhtar/Desktop/revision/disobind/splitting/'

# Load the data
cluster_df = pd.read_csv(base_path+'clustered_0.3.tsv', sep='\t', header=None, names=['representative', 'individual'])
benchmarksZDOCK = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_zdock.txt', dtype=np.dtype('U')))))
benchmarks53 = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_53.txt', dtype=np.dtype('U')))))
benchmarks60 = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_60.txt', dtype=np.dtype('U')))))
benchmarksMFIB = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_MFIB_excluded.txt', dtype=np.dtype('U')))))
benchmarksIDR = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_idrbind.txt', dtype=np.dtype('U')))))
benchmarksFuzDB = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_fuzdb_excluded.txt', dtype=np.dtype('U')))))
benchmarksBret42 = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_bret42.txt', dtype=np.dtype('U')))))
benchmarksDisoBind = np.array(list(map(lambda x: x[:4].upper() + x[4:], np.genfromtxt(base_path+'ids_disobind.txt', dtype=np.dtype('U')))))


# MFIB, FuzDB and ZDock are too big to be in test set entirely
#_, benchmarksMFIB = train_test_split(benchmarksMFIB, test_size=0.2, random_state=42)
#_, benchmarksZDOCK = train_test_split(benchmarksZDOCK, test_size=0.4, random_state=42)
#_, benchmarksFuzDB = train_test_split(benchmarksFuzDB, test_size=0.2, random_state=42)


#benchmarks = np.concatenate((benchmarksDisoBind, benchmarksBret42, benchmarksFuzDB, benchmarksIDR, benchmarksMFIB, benchmarks60, benchmarks53, benchmarksZDOCK))
#benchmarks = np.concatenate((benchmarks60, benchmarks53, benchmarksZDOCK))
benchmarks = np.concatenate((benchmarksBret42, benchmarksIDR, benchmarksMFIB, benchmarksFuzDB))
#benchmarks = benchmarksFuzDB
print (len(benchmarks))


# Create clusters from tsv
clusters = {}
for _, row in cluster_df.iterrows():
    rep = row['representative']
    ind = row['individual']
    if rep not in clusters:
        clusters[rep] = set()
    clusters[rep].update([rep, ind]) 


train_clusters=[]
valid_extension_entries=[]
for rep,list1 in clusters.items():
    if any(item in list1 for item in benchmarks):
        valid_extension_entries.extend(list1)
    else:
        train_clusters.append(rep)

# Split the training into train + valid and extend to all members of clusters
train_clusters, valid_clusters = train_test_split(train_clusters, test_size=0.02, random_state=42) #

train_entries = [item for rep in train_clusters for item in clusters[rep]]
valid_entries = [item for rep in valid_clusters for item in clusters[rep]]


# OPTIONAL: extend validation set with a fraction of extension
valid_extension_entries = [i for i in valid_extension_entries if i not in benchmarks]
#_, valid_extension_entries = train_test_split(valid_extension_entries, test_size=0.5, random_state=42)
valid_entries += valid_extension_entries
print (len(valid_extension_entries))

print (f"Training set --> {len(train_entries)} entries")
print (f"Validation set --> {len(valid_entries)} entries")
assert not any(i in train_entries for i in benchmarks), "Benchmark leaked into training!"

# Save the results to files
train_entries = list(set(train_entries))
with open('splits/train_justIDR.txt', 'w') as f:
    for item in train_entries:
        f.write(f"{item}\n")
valid_entries = list(set(valid_entries))
with open('splits/valid_justIDR.txt', 'w') as f:
    for item in valid_entries:
        f.write(f"{item}\n")
