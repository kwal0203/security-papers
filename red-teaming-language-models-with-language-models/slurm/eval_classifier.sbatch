#!/bin/bash
#SBATCH --job-name=eval-classifier
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=runs/%x-%j.log

# Submit from the repo root, with runs/ already created:
#   sbatch slurm/eval_classifier.sbatch
#   CKPT=runs/roberta-large sbatch slurm/eval_classifier.sbatch
# Pin a GPU type your torch build supports. The cluster's torch has no kernels for V100s
# (compute capability 7.0), so a plain gpu:1 can land on one and crash. gpu:l40s:1 also works.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PY="$HOME/scratch/venvs/red-teaming-language-models-with-language-models/bin/python"
DATA="data/bad/bot_adversarial_dialogue_datasets_with_persona"
CKPT="${CKPT:-runs/roberta-base}"

echo "host: $(hostname)  job: $SLURM_JOB_ID  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" eval_classifier.py --data "$DATA" --ckpt "$CKPT" "$@"

echo "done: $(date)"
