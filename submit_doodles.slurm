#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=test_out.txt
#SBATCH --error=test_error.txt
#
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --gres=gpu:titanxp:1

module load cuda

CUDA_VISIBLE_DEVICES=1 /home/epeake/venv/mainenv/bin/python3 -u mike_test.py -W
