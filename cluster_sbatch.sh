#!/usr/bin/env bash
#SBATCH -J bigmaq
#SBATCH -p a100-galvani
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH --array=0
#SBATCH --output=/mnt/lustre/work/giese/gtc913/logs/BigMaQ/opt-%A_%a.out
#SBATCH --error=/mnt/lustre/work/giese/gtc913/logs/BigMaQ/opt-%A_%a.err

set -euxo pipefail

PYTHON="/mnt/lustre/work/giese/gtc913/conda_envs/monkey_copy/bin/python"
SCRIPT="/mnt/lustre/work/giese/gtc913/pycharm/BigMaQ/pose_reconstruction/scripts/ActionTracking.py"

echo "Worker: $SLURM_ARRAY_TASK_ID"

srun "$PYTHON" -u "$SCRIPT" \
  --worker "$SLURM_ARRAY_TASK_ID" \
  --compute-cfg Setup_Cluster_cfg.json