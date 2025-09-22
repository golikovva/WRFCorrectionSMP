#!/bin/bash

#SBATCH --job-name=wrf_corr                  # Job name
#SBATCH --partition=gpu                      # Queue name
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=20                   # CPU cores per task
#SBATCH --mem=20000                          # Memory in MB
#SBATCH --time=24:00:00                      # Wall time
#SBATCH --output=/storage/thalassa/users/vgolikov/logs/pytorch-test-output.log
#SBATCH --gres=gpu:1                         # <-- request 1 GPU (not 0!)


# --- Run containerized job ---
apptainer exec --nv \
  --pwd /home/experiments/train_test \
  --bind $(pwd):/home \
  --bind /storage/thalassa/users/vgolikov/stations:/home/stations \
  --bind /storage/thalassa/users/vgolikov/borey_scatter:/home/scatter \
  --bind /storage/thalassa/users/vgolikov/logs:/home/logs \
  --bind /storage/thalassa/SMP/WRF_OUTPUT/v2:/home/wrf_smp_data \
  --bind /storage/thalassa/DATA/ERA5:/home/era_global_data \
  /storage/thalassa/users/vgolikov/wrf_correction_smp.sif \
  python main.py
