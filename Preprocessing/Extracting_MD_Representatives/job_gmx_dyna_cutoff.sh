#!/bin/bash
#SBATCH --output=job_%A_%a.out
#SBATCH --constraint=cpu
#SBATCH --mem=64G
####SBATCH --ntasks-per-node=1
###SBATCH --nodelist=sb-node[1-4]

source miniconda3/etc/profile.d/conda.sh
conda activate md

set -ue

gromacs_path="/softs/gromacs/2024.2/bin"
echo -e "c\nc\nc" | $gromacs_path/gmx trjcat -f prot_r*.xtc -o merged_trajectory.xtc -cat -settime

# Generate index
echo -e "3\nq" | $gromacs_path/gmx make_ndx -f prot.pdb -o cindex.ndx
# Calculate RMSD
echo "3 3" | $gromacs_path/gmx rms -s prot.pdb -f merged_trajectory.xtc -n cindex.ndx -o rmsd.xvg -tu ns -m rmsd-matrix.xpm

# Initial cutoff values
cutoff=0.15
max_clusters=50
min_clusters=15

while true; do
    # Run clustering
    echo -e "3\n1" | $gromacs_path/gmx cluster -s prot.pdb -f merged_trajectory.xtc -dm rmsd-matrix.xpm -o clusters.xpm -g cluster.log -method gromos -cl clusters.pdb -cutoff $cutoff
    # Count the number of clusters
    num_clusters=$(grep "Found" cluster.log | awk '{print $2}')

    if [[ $num_clusters -le $max_clusters && $num_clusters -ge $min_clusters ]]; then
        break
    elif [[ $num_clusters -lt $min_clusters ]]; then
        cutoff=$(echo "$cutoff - 0.01" | bc)
    elif [[ $num_clusters -gt $max_clusters ]]; then
        cutoff=$(echo "$cutoff + 0.01" | bc)
    fi

    # Prevent cutoff from going below a certain threshold or above a reasonable value
    if (( $(echo "$cutoff < 0.05" | bc -l) )); then
        echo "Cutoff too low, clustering failed to converge within the desired range."
        break
    elif (( $(echo "$cutoff > 1" | bc -l) )); then
        echo "Cutoff too high, clustering failed to converge within the desired range."
        break
    fi
done

echo "Clustering completed with cutoff $cutoff and $num_clusters clusters."
