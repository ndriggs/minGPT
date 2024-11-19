#!/bin/bash --login

#SBATCH --time=06:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --output=projects/project2a/finetuning_%A_%a.out
#SBATCH --error=projects/project2a/finetuning_%A_%a.err
#SBATCH --mem-per-cpu=102400M   # memory per CPU core
#SBATCH -J "finetuning"   # job name
#SBATCH --array=0-2
#SBATCH --qos=dw87

checkpoints=(74000 111000 148000)

checkpoints_index=$((SLURM_ARRAY_TASK_ID % ${#checkpoints[@]}))

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate mingpt_env
python -u -m projects.project2a.fine_tune --starting_iter=${checkpoints[$checkpoints_index]}