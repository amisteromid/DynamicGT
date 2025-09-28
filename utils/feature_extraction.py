"""
feature_extraction.py: Compute and extract dynamic features 
Omid Mokhtari - Inria 2025
This file is part of DynamicGT.
Released under CC BY-NC-SA 4.0 License
"""

import numpy as np
from scipy.stats import entropy
    

def extract_dynamic_features(xyz, nn_topk):
	'''
	1) Dynamic Correlation:
		1.1) vector: motion_v
		1.2) scalar: motion_s
	2) Root Mean Square Fluctuation (RMSF1) - True Fluctuation 
	3) Root Mean Square Fluctuation (RMSF2) - Directional Entropy
	4) Communication propensity (CP)
	'''
	epsilon = 1e-15
	# Make normalized displacement vectors
	mean_pos = np.mean(xyz, axis=0)
	displacements = xyz - mean_pos
	norms = np.linalg.norm(displacements, axis=-1, keepdims=True)
	normalized_displacements = displacements / norms
	
	num_conformations, num_elements, _ = normalized_displacements.shape
	knn = nn_topk.shape[1]
	
	# Motion vector and scalar
	cross_products = np.zeros((num_conformations, num_elements, knn, 3))
	for i in range(num_elements):
		for j, neighbor in enumerate(nn_topk[i]):
			cross_products[:, i, j, :] = np.cross(normalized_displacements[:, i, :], normalized_displacements[:, neighbor, :])
	cross_products = cross_products.transpose(1,2,0,3)
	motion_v = np.mean(cross_products, axis=2)
	
	# Normalize
	cross_products_magnitude = np.linalg.norm(cross_products, axis=-1, keepdims=True)
	motion_v_magnitude = np.linalg.norm(motion_v, axis=-1, keepdims=True)
	normalized_cross_products = cross_products / (cross_products_magnitude + epsilon)
	normalized_motion_v = motion_v / (motion_v_magnitude + epsilon)
	# Calculate cosine similarity
	cosine_similarity = np.sum(normalized_cross_products * normalized_motion_v[:, :, np.newaxis, :], axis=-1)
	motion_s = np.mean(cosine_similarity, axis=2)
	motion_s_min = motion_s.min(axis=-1, keepdims=True)
	motion_s_max = motion_s.max(axis=-1, keepdims=True)
	motion_s_normalized = (motion_s - motion_s_min) / (motion_s_max - motion_s_min)
	
	# RMSF1 and RMSF2 (DE)
	rmsf1 = np.sqrt(np.mean(np.sum(displacements**2, axis=-1), axis=0))
	normalized_rmsf1 = (rmsf1 - np.min(rmsf1)) / (np.max(rmsf1) - np.min(rmsf1))
	U, sigma, Vt = np.linalg.svd(np.transpose(normalized_displacements[:,:,:], (1, 0, 2)), full_matrices=False)
	sigma = sigma / sigma.sum(axis=1, keepdims=True)
	rmsf2 = np.apply_along_axis(entropy, 1, sigma)
	normalized_rmsf2 = (rmsf2 - np.min(rmsf2)) / (np.max(rmsf2) - np.min(rmsf2))
	
	# Communication propensity (CP)
	dists= np.zeros((num_conformations, num_elements, knn))
	for i in range(num_elements):
		for j, neighbor in enumerate(nn_topk[i]):
			dists[:, i, j] = np.linalg.norm(xyz[:, i, :] - xyz[:, neighbor, :], axis=-1)
	CP = np.var(dists, axis=0)
	normalized_CP = (CP - np.min(CP)) / (np.max(CP) - np.min(CP))
	
	return motion_v, motion_s_normalized, normalized_rmsf1, normalized_rmsf2, normalized_CP

def encode_sequence(seq, list_aa):
	m = (np.array([i for i in seq]).reshape(-1,1) == np.array(list_aa).reshape(1,-1))
	m_plusone =  np.concatenate([m, ~np.any(m, axis=1).reshape(-1,1)], axis=1) # another row to for aa that does not exist in list_aa
	return m_plusone 


def mean_coordinates(xyz):
	# for MD we used mean, but for AF first model
	mean_xyz = xyz[0]
	#mean_xyz = np.mean(xyz, axis=0)
	return mean_xyz

def extract_topology(mean_xyz):
	'''
	1) Pairwise Displacement Vectors (R)
	2) pairwise Distances (D)
	'''
	
	R = mean_xyz[np.newaxis, :, :] - mean_xyz[ :, np.newaxis, :]
	D = np.linalg.norm(R, axis=2)
	
	epsilon = 1e-15
	R_normalized = R / (D[:, :, np.newaxis] + epsilon)
	
	return R_normalized, D
