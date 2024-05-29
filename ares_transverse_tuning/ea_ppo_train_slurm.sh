#!/bin/sh
#SBATCH --partition=maxcpu
#SBATCH --job-name ea-ppo
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=75F3
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate ares-transverse-tuning
cd /beegfs/desy/user/kaiserja/ares-transverse-tuning

python -m src.train.ea_ppo

exit
