#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name ea-ppo
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=P100|V100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate ares-transverse-tuning
cd /beegfs/desy/user/kaiserja/ares-transverse-tuning

wandb agent --count 1 msk-ipc/ares-ea-v3/d2z5rw4e

exit
