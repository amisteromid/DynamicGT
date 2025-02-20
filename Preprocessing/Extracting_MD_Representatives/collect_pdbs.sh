#!/bin/bash

input_dir="/users/omokhtar/PDBbind" # folder of subfolders
output_dir="/users/omokhtar/representatives"

mkdir -p "$output_dir"

for folder in "$input_dir"/*; do
  if [ -d "$folder" ]; then
    folder_name=$(basename "$folder")
    
    src_file="$folder/analysis/clusters.pdb"
    dest_file="$output_dir/${folder_name}.pdb"
    
    if [ -f "$src_file" ]; then
      cp "$src_file" "$dest_file"
    fi
  fi
done

