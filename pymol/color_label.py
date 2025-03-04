import pymol
from pymol import cmd, stored

def create_custom_spectrum(selection="b", name="custom_spectrum", minimum=0, maximum=1):
    # Define the colors
    blue = [0.1, 0.1, 0.9]
    middle = [0.3, 0.3, 0.3]
    red = [0.9, 0.1, 0.1]
    
    colors = []
    steps = 100  # Number of interpolation steps for smooth transition
    
    # Interpolate blue to middle
    for i in range(steps):
        t = i / (steps - 1)
        color = [
            blue[0] + (middle[0] - blue[0]) * t,
            blue[1] + (middle[1] - blue[1]) * t,
            blue[2] + (middle[2] - blue[2]) * t
        ]
        colors.append(color)
    
    # Interpolate middle to red
    for i in range(steps):
        t = i / (steps - 1)
        color = [
            middle[0] + (red[0] - middle[0]) * t,
            middle[1] + (red[1] - middle[1]) * t,
            middle[2] + (red[2] - middle[2]) * t
        ]
        colors.append(color)
    for i, color in enumerate(colors):
        color_name = f"{name}_{i}"
        cmd.set_color(color_name, color)
    cmd.spectrum(selection, name, minimum=minimum, maximum=maximum)

create_custom_spectrum("b", "blue_gray_red", minimum=0, maximum=1)

for i in cmd.get_names():
	#pdb,chain,_ = i.split('_')
	cmd.remove("not polymer")
	# Make more contrast with certain threshold
	cmd.alter(f"{i} and b > 0.5", "b = 1 - (1 - b) * 0.9")
	cmd.alter(f"{i} and b < 0.5001", "b = b * 0.9")

	blue0 = [0.1, 0.1, 0.9]
	middle = [0.1, 0.1, 0.1]
	red1 = [0.9, 0.1, 0.1]
	
	cmd.spectrum("b", "blue_gray_red", f'{i}', minimum=0, maximum=1)
	
	#cmd.color('gray20', f'{i} and not chain {chain}')
	#cmd.extract(f"{i}_not_{chain}", f"{i} and not chain {chain}")
	#cmd.set('cartoon_transparency', 0.25, f"{i}_not_{chain}")
	#cmd.set('cartoon_transparency', 0.15, f"{i} and not chain {chain}")
	cmd.show('sphere', )#f'{i} and chain {chain}'
            

#cmd.set('transparency',0.5, 'chain A')




