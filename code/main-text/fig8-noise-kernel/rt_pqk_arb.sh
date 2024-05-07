#!/bin/bash
# Execution wrapper script for Bose cluster
# Combine stdout and sterr into one output file:
#$ -j y
# Use "bash" shell:
#$ -S /bin/bash
# Use current directory as working root:
#$ -cwd
#PBS -N pqk_expressibility
#PBS -P Personal
#PBS -l select=1:ncpus=1
#PBS -l walltime=1:00:00
#PBS -o sge_output/
#PBS -e sge_output/
cd ${PBS_O_WORKDIR}
module load singularity
singularity exec /app/singularity/images/pennylane.sif python pqk_noise_arb.py $qubit $layer_tot $name_data $sample $batch $noise

