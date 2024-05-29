#!/bin/sh
#SBATCH --partition=single
#SBATCH --job-name ea-tune-ucb
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-type ALL
#SBATCH --output=slurm_logs/training_%j.out



source ~/.bashrc
conda activate ares-transverse-tuning

# Move in Workspace to avoid I/O in cluster HOME
rsync -a ~/ares-transverse-tuning $(ws_find ares-transverse)/
cd $(ws_find ares-transverse)/ares-transverse-tuning

# Start Sweep
wandb agent --count 1 msk-ipc/ares-opttune/2oeonv9o

exit