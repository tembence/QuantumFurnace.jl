#!/bin/bash -l
#SBATCH --job-name=quantum_oven
#SBATCH --output=slurm_out/job-%j.out
#SBATCH --error=slurm_out/job-%j.err

# 1. Set the number of parallel jumps (tasks).
#SBATCH --ntasks=15

# 2. Set the number of cores (threads) for EACH jump's matrix math.
#SBATCH --cpus-per-task=8

# 4. Memory is now per CPU.
#    8 cpus/task * 4GB/cpu = 32GB per task.
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

#==============================================================================
# --- Job Execution ---
#==============================================================================

echo "========================================================="
echo "Job started on $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Allocated $SLURM_NTASKS tasks."
echo "========================================================="

# Create the output directory if it doesn't exist
mkdir -p slurm_out

echo "Loading Julia module..."
module purge
module load Julia/1.10.4-linux-x86_64

export JULIA_DEPOT_PATH="/bigwork/nhbbwebt/.julia"
echo "JULIA_DEPOT_PATH is set to: $JULIA_DEPOT_PATH"

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "JULIA_NUM_THREADS set to: $JULIA_NUM_THREADS"

JULIA_SCRIPT_TO_RUN="src/main_liouv.jl"
# JULIA_SCRIPT_TO_RUN="src/main_thermalize.jl"

echo "Target Julia script: $JULIA_SCRIPT_TO_RUN"

echo "Launching Julia file..."
julia --project=@. "$JULIA_SCRIPT_TO_RUN"  # No srun

echo "========================================================="
echo "Job finished on $(date)"
echo "========================================================="