import os
import glob
from pymol import cmd

####################################################################
############## CHEEEEEEEEEECK line 52 ##############################
####################################################################

parent_dir='/media/omokhtar/Expansion/Omid/AlphaFlow/'
parent_folder='raw/Benchmarks_60_53_IDR_ag_fuzdb'
new_folder='renamed_benchmarks_filtered'
cutoff=3

def get_sequence_length_ratio(sel1, sel2):
    seq1 = cmd.get_fastastr(sel1).split('\n')[1]
    seq2 = cmd.get_fastastr(sel2).split('\n')[1]
    
    len_og, len_af = len(seq2),len(seq1)
    
    ratio = (len_af / len(len_og)) * 100
    
    return ratio
    

ry
target_files = glob.glob(parent_dir+f'{parent_folder}/*.pdb')

for target_file in target_files:
    try:
        cmd.reinitialize()
        target = os.path.basename(target_file).split('.')[0]
        cmd.load(target_file, target)
    
        # If structure.pdb has one chain
        cmd.fetch(f'{target[:4]}') #, type='pdb1'
        cmd.remove('not polymer.protein')

        try:
            rms = cmd.super(f'{target}', f'{target[:4]} and chain {target[5:]}')[0]
        except:
            rms=99999
        try:
            seq_length_ratio = get_sequence_length_ratio(f'{target}', f'{target[:4]} and chain {which_chain}')
        except:
            seq_length_ratio=0
        with open(parent_dir+'bad_rmsd.txt','a') as file:
            file.write(f"{target} {seq_length_ratio} {rms}\n")
        
        if rms<cutoff:
            cmd.save(parent_dir+f'{new_folder}/{target[:4]}_{target[5:]}.pdb', target, state=0)
        else:
            # !!!!!!!!!!!!!! Save even if not good rmsd !!!!!!!!!!!!
            cmd.save(parent_dir+f'{new_folder}/{target[:4]}_{target[5:]}.pdb', target, state=0)
            
    except Exception as e:
        with open(parent_dir + 'error_log.txt', 'a') as error_file:
            error_file.write(f"Error with target {target}: {str(e)}\n")

