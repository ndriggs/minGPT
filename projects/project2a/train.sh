#!/bin/bash --login

#SBATCH --time=00:30:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --output=pre_training.out
#SBATCH --error=pre_training.err
#SBATCH --mem-per-cpu=102400M   # memory per CPU core
#SBATCH -J "initial_training"   # job name
#SBATCH --qos=dw87

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate mingpt_env
python -u train_on_pile.py