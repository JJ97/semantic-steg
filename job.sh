#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p gpu-small
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --job-name=steg

module load cuda/9.0  
source ../steg/bin/activate
cd prep/twitter
stdbuf -oL python3 main.py
