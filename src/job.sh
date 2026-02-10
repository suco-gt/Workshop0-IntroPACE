#!/bin/bash
#SBATCH --job-name=matmul_extralarge

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=1:00:00

#SBATCH --output=matmul_%j.out
#SBATCH --error=matmul_%j.err

# Clean environment
module purge
module load gcc/12.3.0
module load mvapich2/2.3.7-1

# Go to directory where you ran sbatch
cd $SLURM_SUBMIT_DIR

# Compile + run extralarge matrix
make run MPI_LAUNCH="srun" MATRIX_SIZE=16384