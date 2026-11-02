#!/usr/bin/env -S nix develop .# -c bash
#SBATCH --time=02:00:00
#SBATCH --job-name=mi_gpujob
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
export AUDIOINTERP_CONFIG="$SLURM_SUBMIT_DIR/config.local.yaml"
export HF_HOME=/data/chi-gpu1/go97xis/.hf_cache
export PYTHONUNBUFFERED=1

.venv/bin/python -m "$@"
