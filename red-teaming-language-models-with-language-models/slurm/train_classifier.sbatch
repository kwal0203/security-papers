#!/bin/bash
#SBATCH --job-name=train-classifier
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.log

# Submit from the repo root, with runs/ already created (Slurm won't create the log dir):
#   mkdir -p runs
#   sbatch slurm/train_classifier.sbatch
#   sbatch slurm/train_classifier.sbatch --epochs 5            # extra args go to the script
#   MODEL=roberta-large OUT=runs/roberta-large sbatch slurm/train_classifier.sbatch
# MODEL/OUT reach the job because sbatch exports your environment by default.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PY="$HOME/scratch/venvs/red-teaming-language-models-with-language-models/bin/python"
DATA="data/bad/bot_adversarial_dialogue_datasets_with_persona"
MODEL="${MODEL:-roberta-base}"
OUT="${OUT:-runs/roberta-base}"

echo "host: $(hostname)  job: $SLURM_JOB_ID  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" train_classifier.py --data "$DATA" --model "$MODEL" --out "$OUT" "$@"

echo "done: $(date)"
