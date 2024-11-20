#!/bin/bash --login

#SBATCH --time=00:55:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --output=projects/project2a/evaluation_%A_%a.out
#SBATCH --error=projects/project2a/evaluation_%A_%a.err
#SBATCH --mem-per-cpu=102400M   # memory per CPU core
#SBATCH -J "evaluation"   # job name
#SBATCH --array=0-9
#SBATCH --qos=dw87

checkpoints=("37000" "74000" "74000_6730" "74000_13460" "111000" "111000_6730" "111000_13460" "148000" "148000_6730" "148000_13460")

checkpoints_index=$((SLURM_ARRAY_TASK_ID % ${#checkpoints[@]}))

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate mingpt_env
python projects/project2a/evaluate.py ${checkpoints[$checkpoints_index]}