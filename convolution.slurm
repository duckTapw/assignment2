#!/bin/bash

#SBATCH --job-name=mpi-convolution
#SBATCH --output=convolution.out
#SBATCH --error=convolution.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=23086203@student.uwa.edu.au
#SBATCH --partition=cits3402

# Load MPI
module load gcc/14.3
module load openmpi/5.0.5

make
./convolution -H 25000 -W 25000 -kH 27 -kW 27
./convolution_omp -H 30000 -W 30000 -kH 30 -kW 30
mpiexec -n 4 ./convolution_mpi -H 20000 -W 20000 -kH 25 -kW 25
#mpiexec -n 4 ./convolution_mpi -H 10000 -W 10000 -kH 25 -kW 25