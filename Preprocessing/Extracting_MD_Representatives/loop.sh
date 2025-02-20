#!/bin/bash

for dir in /users/omokhtar/PDBbind/*/analysis

do
    cd "$dir"
    sbatch /users/omokhtar/job_gmx_dyna_cutoff.sh
    cd /users/omokhtar/PDBbind/
done

