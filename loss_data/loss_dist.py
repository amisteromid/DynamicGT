"""
loss_dist.py: Calculating geometric distances for the loss function
Omid Mokhtari - Inria 2025
This file is part of DynamicGT.
Released under CC BY-NC-SA 4.0 License
"""

import subprocess
import os
import numpy as np
import requests
from scipy.spatial import cKDTree
from Bio import PDB
import networkx as nx
from Bio.Data.SCOPData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio import pairwise2

def fetch_and_extract_chain(pdb_id, chain, folder='/home/omokhtar/Desktop/alphaflow/results/benchmarks'):
    file_path = os.path.join(folder, f"{pdb_id}_{chain}.pdb")
    if os.path.exists(file_path):
        return file_path

def extract_seq_and_calpha_coords(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    sequence = []
    calpha_coords = []
    
    # Iterate through each model in the structure
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    calpha = residue['CA']
                    calpha_coords.append(calpha.coord)
                    
                    res_name = residue.get_resname()
                    sequence.append(protein_letters_3to1.get(res_name, 'X'))
    
    if len(sequence) != len(calpha_coords):
        raise ValueError(f"Length mismatch: sequence length ({len(sequence)}) does not match C-alpha coordinates length ({len(calpha_coords)})")
    
    return ''.join(sequence), calpha_coords

    
def parse_verts(vert_file, face_file, keep_normals=False):
    """
    Generate the vertices and faces from .vert and .face files generated by MSMS
    """
    with open(vert_file, 'r') as f:
        lines = f.readlines()
        n_vert = int(lines[2].split()[0])
        no_header = lines[3:]
        assert len(no_header) == n_vert
        
        lines = [line.split()[:6] for line in no_header]
        lines = np.array(lines).astype(np.float32)
        verts = lines[:, :3]
        if keep_normals:
            normals = lines[:, 3:6]

    with open(face_file, 'r') as f:
        lines = f.readlines()
        n_faces = int(lines[2].split()[0])
        no_header = lines[3:]
        assert len(no_header) == n_faces

        lines = [line.split() for line in no_header]
        lines = np.array(lines).astype(np.int32)
        faces = lines[:, :3]
        faces -= 1

    if keep_normals:
        return verts, faces, normals
    else:
        return verts, faces
        
def pdb_to_surf(pdb, out_name, density=1., clean_temp=True):
    """
    Runs msms on the input PDB file and dumps the output in out_name
    """
    print ("start with ---->",out_name)
    vert_file = out_name + '.vert'
    face_file = out_name + '.face'

    # First get the xyzr file
    temp_xyzr_name = f"{out_name}_temp.xyzr"
    temp_log_name = f"{out_name}_msms.log"
    with open(temp_xyzr_name, "w") as f:
        cline = f"/home/omokhtar/Desktop/dist_loss/msms_i86_64Linux2_2.6.1/pdb_to_xyzr {pdb}"
        subprocess.run(cline.split(), stdout=f)

    # Then run msms on this file
    cline = f"/home/omokhtar/Desktop/dist_loss/msms_i86_64Linux2_2.6.1/msms -if {temp_xyzr_name} -of {out_name} -density {density}"
    with open(temp_log_name, "w") as f:
        result = subprocess.run(cline.split(), stdout=f, stderr=f, timeout=10)
    if result.returncode != 0:
        print(f"*** An error occurred while executing the command: {cline}, see log file for details. *** ")
        raise RuntimeError(f"MSMS failed with return code {result.returncode}")

    if clean_temp:
        os.remove(temp_xyzr_name)
    os.remove(temp_log_name)

    verts, faces = parse_verts(vert_file=vert_file, face_file=face_file)
    os.remove(vert_file)
    os.remove(face_file)

    return verts, faces
    

def assign_calpha_to_vertices(calpha_coords, verts):
    calpha_coords = np.array(calpha_coords)
    verts = np.array(verts)
    
    # KDTree for nearest-neighbor search
    tree = cKDTree(verts)
    
    # Find the closest vertex for each C-alpha
    distances, assigned_indices = tree.query(calpha_coords)
    
    return assigned_indices, np.array(distances)
    
    
def create_graph(verts, faces):
    # Create an undirected graph
    G = nx.Graph()
    
    # Add vertices with their positions
    for i, vert in enumerate(verts):
        G.add_node(i, pos=vert)
    
    # Add edges based on faces
    for face in faces:
        G.add_edge(face[0], face[1])
        G.add_edge(face[1], face[2])
        G.add_edge(face[2], face[0])
    
    return G

def compute_shortest_paths(graph, ca_euclidean_distances, calpha_indices, labels):
    size = len(calpha_indices)
    distances = np.full(size, float('inf'))  # Initialize distances with infinity
    
    for i, n in enumerate(calpha_indices):
        if labels[i] == 1:  # If the current label is 1, set distance to 0
            distances[i] = 0
        else:
            if n == -1:  # If the current node is -1 (core residue)
                distances[i] = min([d for pos,d in enumerate(ca_euclidean_distances[i, :]) if labels[pos]==1])
            else: # not core and not IF
                for ii, nn in enumerate(calpha_indices):
                    if labels[ii] == 1 and nn != -1:
                        tmp_d = nx.shortest_path_length(graph, source=n, target=nn)
                        if tmp_d < distances[i]:
                            distances[i] = tmp_d
    
    return distances
    
def normalize(dists, ca_vert_indices):
    # Identify core and surface indices
    cores = np.where(ca_vert_indices == -1)[0]
    surfaces = np.where(ca_vert_indices != -1)[0]
    
    if all(value == 0 for value in dists):
        return dists
    else:
        if len(cores) > 0:
            dists[cores] = ((dists[cores]-np.min(dists[cores])) / (np.max(dists[cores])-0))
        if len(surfaces) > 0:
            dists[surfaces] = (dists[surfaces]-np.min(dists[surfaces])) / (np.max(dists[surfaces])-np.min(dists[surfaces]))
    return dists
	
def get_dist(target, chain, labels, seq1):
	target = fetch_and_extract_chain(target, chain)
	seq2, ca_coords = extract_seq_and_calpha_coords(target)
	verts, faces = pdb_to_surf(target, target.split('/')[-1][:-4], density=1., clean_temp=True)
	
	# Construct graph from mesh
	graph = create_graph(verts, faces)
	
	# Assign C-alpha atoms to closest vertices
	ca_vert_indices, ca_vert_distances = assign_calpha_to_vertices(ca_coords, verts)
	ca_euclidean_distances = np.sqrt(np.sum((np.array(ca_coords)[:, np.newaxis, :] - np.array(ca_coords)[np.newaxis, :, :])**2, axis=-1)) # n_atom * n_atom matrix
	# core C-alphas don't have any vertice correspondence
	ca_vert_indices[np.where(ca_vert_distances>6)[0]] = -1
	# Compute shortest path distances between C-alpha atoms
	if len(labels)==len(ca_vert_distances):
		shortest_path_distances = compute_shortest_paths(graph, ca_euclidean_distances, ca_vert_indices, labels)
		dists = normalize(shortest_path_distances, ca_vert_indices)
	elif len(labels)<len(ca_vert_distances):
		alignment = pairwise2.align.globalms(seq1, seq2, 2, -1, -10, -0.5)[0]
		seq1_aligned, seq2_aligned = alignment[0], alignment[1]
		assert seq2_aligned.count('-')==0,f"{seq1_aligned}\n{seq2_aligned}"
		assert len(seq2_aligned)==len(seq2),f"{seq2}\n{seq2_aligned}"
		remove_set = [i for i,n in enumerate(seq1_aligned) if n=='-']
		keep_indices = [i for i in range(len(seq2)) if i not in remove_set]
		ca_euclidean_distances =  ca_euclidean_distances[np.ix_(keep_indices, keep_indices)]
		ca_vert_indices = ca_vert_indices[keep_indices]
		shortest_path_distances = compute_shortest_paths(graph, ca_euclidean_distances, ca_vert_indices, labels)
		dists = normalize(shortest_path_distances, ca_vert_indices)
	else:
		print (f"Label and distance matrix do not match in size!\n{len(labels)},{target}_{chain}")
	
	return dists

if __name__ == "__main__":
	dists = get_dist('1A5K', 'C', [],'sss')
