#!/bin/bash
#SBATCH --job-name=train_sg
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=6                                # Ask for 2 CPUs
#SBATCH --gres=gpu:a100:1                                     # Ask for 1 GPU
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM
#SBATCH --time=03:50:00 

module load cudatoolkit/11.7
source ~/venv/bin/activate



model=$1
size=$2
cache_dir='cache_heldout_sg_'$model'_'$size'/'
model_name=$3 #'google/flan-t5-large'
output_dir='model_ckpts_heldout_sg/'$model'_'$size


bash ds_train_sg.sh $cache_dir $model_name $output_dir

