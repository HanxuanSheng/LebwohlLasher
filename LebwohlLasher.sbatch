#!/bin/bash
# This tells SLURM to use the bash shell

#SBATCH --job-name=LebwohlLashe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --account=phys033186
#SBATCH --partition=teach_cpu

module load python/3.11
module load mpi         

mpirun -np 4 python mpi_LebwohlLasher.py 10000 100 0.5 0
