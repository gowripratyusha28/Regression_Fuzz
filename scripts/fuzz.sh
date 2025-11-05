#!/bin/bash
#SBATCH --account=swabhas_1625     
#SBATCH --partition=nlp
#SBATCH --time=1-10:00:00                  
#SBATCH --mem=64GB                        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                
#SBATCH --gpus-per-task=2
#SBATCH --job-name=fuzz
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate nyc-connections
export VLLM_USE_FLASH_ATTENTION=0

export HF_HOME=/project2/swabhas_1716/hf_home
mutator_model=meta-llama/Llama-3.1-8B-Instruct
# mutator_model=mistralai/Mistral-Nemo-Instruct-2407
target_model=meta-llama/Llama-3.1-8B-Instruct


time python run.py --model_path ${mutator_model} --target_model ${target_model}