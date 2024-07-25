#!/bin/bash
#SBATCH --job-name=train_act_sg
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=6                                # Ask for 2 CPUs
#SBATCH --gres=gpu:a100:1                                     # Ask for 1 GPU
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM
#SBATCH --time=03:50:00                                   # The job will run for 3 hours



module load cudatoolkit/11.7
source ~/venv/bin/activate

model=$1
size=$2
cache_dir='cache_heldout_action_sg_'$model'_'$size'/'
model_name=$3 #'google/flan-t5-large'
output_dir='model_ckpts_heldout_action_sg/'$model'_'$size

echo $cache_dir $model_name $output_dir

bash ds_train.sh $cache_dir $model_name $output_dir

