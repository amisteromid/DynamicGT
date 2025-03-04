from pymol import cmd
import os
import time

structures = ['8G7I_A']
'''
structures = [
    '8G7I_B_io0_out', '8G7I_B_io1_out', '8G7I_B_io2_out', '8G7I_B_io3_out', '8G7I_B_io4_out',
    '8G7I_B_io5_out', '8G7I_B_io6_out', '8G7I_B_io7_out', '8G7I_B_io8_out', '8G7I_B_io9_out', 
    '8G7I_B_io10_out', '8G7I_B_io11_out', '8G7I_B_io12_out', '8G7I_B_io13_out', '8G7I_B_io14_out', 
    '8G7I_B_io15_out', '8G7I_B_io16_out', '8G7I_B_io17_out', '8G7I_B_io18_out', '8G7I_B_io19_out', 
    '8G7I_B_io20_out', '8G7I_B_io21_out', '8G7I_B_io22_out', '8G7I_B_io23_out', '8G7I_B_io24_out', 
    '8G7I_B_io25_out', '8G7I_B_io26_out', '8G7I_B_io27_out', '8G7I_B_io28_out', '8G7I_B_io29_out', 
    '8G7I_B_io30_out', '8G7I_B_io31_out', '8G7I_B_io32_out', '8G7I_B_io33_out', '8G7I_B_io34_out'
]
'''
# number of conformations per structure
num_conformations = 25

# output directory
output_dir = "pymol_movie_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cmd.center()
cmd.zoom(complete=1)


frame_num = 1
try:
    for structure in structures:
        cmd.show("surface", structure)
    cmd.center()
    cmd.zoom()
    time.sleep(1)  # Brief pause
    
    cmd.hide("everything")
    
    # Process each structure
    for i, structure in enumerate(structures):
        print(f"Processing structure {i+1}/{len(structures)}: {structure}")
        
        cmd.show("surface", structure)
        cmd.center(structure)
        cmd.zoom(structure, 5)
        
        # Loop through each conformation
        for conformation in range(1, num_conformations + 1):
            cmd.frame(conformation)
            cmd.center(structure)
            filename = f"{output_dir}/frame_{frame_num:04d}.png"
            cmd.ray(800, 600)
            cmd.png(filename, ray=1, quiet=1)
            print(f"Saved frame {frame_num}: {filename}")
            frame_num += 1
        cmd.hide("everything", structure)
    print("Use FFmpeg to create a movie from these frames:")
    print("ffmpeg -framerate 25 -i " + output_dir + "/frame_%04d.png -c:v libx264 -vf 'pad=1630:910:0:0:black' -pix_fmt yuv420p movie2.mp4")
except Exception as e:
    print(f"An error occurred: {e}")
