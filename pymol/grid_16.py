from pymol import cmd


#cmd.split_states('1UEL_A_updated')
#cmd.extra_fit()

structures = ['2KHS_A_updated_0001', '2KHS_A_updated_0002', '2KHS_A_updated_0003', '2KHS_A_updated_0004', '2KHS_A_updated_0005', '2KHS_A_updated_0006', '2KHS_A_updated_0007', '2KHS_A_updated_0008', '2KHS_A_updated_0009', '2KHS_A_updated_0010', '2KHS_A_updated_0011', '2KHS_A_updated_0012', '2KHS_A_updated_0013', '2KHS_A_updated_0014', '2KHS_A_updated_0015', '2KHS_A_updated_0016']

# Grid parameters
grid_size = 4  # 4x4 grid
spacing = [50, 55, 0]  # Translation between structures

def arrange_grid():
    # Hide all structures initially
    #cmd.hide('everything', '*')
    
    # Show cartoon representation for all structures
    #cmd.show('cartoon', '*')
    
    # Arrange structures in grid
    for i, structure in enumerate(structures[:16]):  # Only take first 16 for 4x4 grid
        row = i // grid_size
        col = i % grid_size
        
        trans_x = col * spacing[0]
        trans_y = row * spacing[1]
        trans_z = 0
        
        cmd.translate([trans_x, trans_y, trans_z], structure)

arrange_grid()

cmd.center('all')
cmd.zoom('all')
