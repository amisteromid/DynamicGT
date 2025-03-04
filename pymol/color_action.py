import pymol
from pymol import cmd, stored
import colorsys
import math

def calculate_color_in(in_value, out_value):
    # Ensure in_value is between 0 and 1
    in_value = max(0, min(1, in_value))
    
    # Define colors
    gray70 = (0.9, 0.9, 0.9)  # RGB for gray70
    deep_blue = (0, 0, 0.5)   # RGB for deep blue
    
    # Interpolate between gray70 and deep blue
    r = gray70[0] + (deep_blue[0] - gray70[0]) * in_value
    g = gray70[1] + (deep_blue[1] - gray70[1]) * in_value
    b = gray70[2] + (deep_blue[2] - gray70[2]) * in_value
    
    return r, g, b

def calculate_color_out(in_value, out_value):
    # Ensure in_value is between 0 and 1
    out_value = max(0, min(1, out_value))
    
    # Define colors
    gray70 = (0.9, 0.9, 0.9)  # RGB for gray70
    deep_red = (0.5, 0, 0)   # RGB for deep red
    
    # Interpolate between gray70 and deep blue
    r = gray70[0] + (deep_red[0] - gray70[0]) * out_value
    g = gray70[1] + (deep_red[1] - gray70[1]) * out_value
    b = gray70[2] + (deep_red[2] - gray70[2]) * out_value
    
    return r, g, b

for i in cmd.get_names():
    pdb, chain, _ = i.split('_')
    model = cmd.get_model(i)
    
    # Create two new objects for in and out coloring
    cmd.create(f"{i}_in", i)
    cmd.create(f"{i}_out", i)
    cmd.show('sphere')
    
    for atom in model.atom:
        b_factor = str(atom.b)
        in_logit, out_logit = b_factor.split('.')
        in_logit = float('0.' + in_logit)
        out_logit = float('0.' + out_logit)
        
        # Color for in_logit
        r_in, g_in, b_in = calculate_color_in(in_logit, out_logit)
        cmd.set_color(f"color_in_{r_in}_{g_in}_{b_in}", [r_in, g_in, b_in])
        cmd.color(f"color_in_{r_in}_{g_in}_{b_in}", f'{i}_in and ID {atom.index}')
        
        # Color for out_logit
        r_out, g_out, b_out = calculate_color_out(in_logit, out_logit)
        cmd.set_color(f"color_out_{r_out}_{g_out}_{b_out}", [r_out, g_out, b_out])
        cmd.color(f"color_out_{r_out}_{g_out}_{b_out}", f'{i}_out and ID {atom.index}')
    '''
    for c in cmd.get_chains(i):
        if c != chain:
            cmd.color('gray30', f'{i}_in and chain {c}')
            cmd.color('gray30', f'{i}_out and chain {c}')
            cmd.extract(f"{i}_{c}_in", f"{i}_in and chain {c}")
            cmd.extract(f"{i}_{c}_out", f"{i}_out and chain {c}")
            cmd.set('cartoon_transparency', 0.35, f"{i}_{c}_out")
            cmd.set('cartoon_transparency', 0.25, f"{i}_{c}_in")
        else:
            cmd.show('surface', f'{i}_in and chain {c}')
            cmd.show('surface', f'{i}_out and chain {c}')
    '''
    
    # Delete the original object
    cmd.delete(i)

