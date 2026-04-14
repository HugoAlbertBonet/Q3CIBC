#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 
#SBATCH --mem=16G
#SBATCH --time=6:00:00

export WANDB_API_KEY=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
srun uv run python hyperparam_search_dfo.py --run --params '{"SOFTMAX_TEMPERATURE": 0.5, "NUM_COUNTER_EXAMPLES": 32, "TRAINING_STEPS": 100000, "LANGEVIN_TRAIN_ITERATIONS": 100, "LR_DECAY_RATE": 0.995}'
