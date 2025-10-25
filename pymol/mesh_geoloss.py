from pymol import cmd
import numpy as np

# Clean start
cmd.hide('everything')
cmd.show('cartoon')

# Color chain A for reference
cmd.color('green', '1F3V_A')

# Get coordinates of all atoms in chain A
stored.coords_A = []
cmd.iterate_state(1, '1F3V_A', 'stored.coords_A.append([x,y,z])')
coords_A = np.array(stored.coords_A)

# For each atom in chain B, find minimum distance to chain A
stored.atoms_B = []
cmd.iterate_state(1, '1F3V_B', 'stored.atoms_B.append((index, x, y, z))')

for idx, x, y, z in stored.atoms_B:
    pos_B = np.array([x, y, z])
    # Calculate distances to all atoms in A
    dists = np.sqrt(np.sum((coords_A - pos_B)**2, axis=1))
    min_dist = np.min(dists)
    
    # Store in b-factor
    cmd.alter(f'1F3V_B and index {idx}', f'b={min_dist}')

# Apply color spectrum: blue (close) to gray80 (far)
cmd.spectrum('b', 'skyblue_lightblue_white', '1F3V_B', minimum=3, maximum=30)

# Show both chains
cmd.show('mesh', '1F3V_A or 1F3V_B')

# Deselect all
cmd.deselect()
