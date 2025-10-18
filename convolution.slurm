#!/bin/bash

#SBATCH --job-name=mpi-convolution
#SBATCH --output=convolution.out
#SBATCH --error=convolution.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=23086203@student.uwa.edu.au
#SBATCH --partition=cits3402

# Load MPI
module load gcc/14.3
module load openmpi/5.0.5

mpicc -o convolution_mpi convolution_mpi.c -fopenmp -lm -O3
mpiexec -n 4 ./convolution_ -H 1000 -W 1000 -kH 5 -kW 5 -o results.txt 

