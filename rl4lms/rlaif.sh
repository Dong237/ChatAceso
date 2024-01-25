export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_CACHE="/scratch/huggingface"
export LD_LIBRARY_PATH="~/miniconda3/envs/rl4lm-v3/lib/python3.9/site-packages/nvidia/cuda_runtime/lib"

# export CUDA_HOME=/usr/local/cuda-11.8
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/wsl/lib
# export PATH=$CUDA_HOME/bin:$PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUBLAS_WORKSPACE_CONFIG=:0:0

python train_rlaif.py \
    --config_path /cluster/home/RL4LMs/config_empathy.yml \
    --project_name "rl4lm-test" \
    --experiment_name "aceso-empadata-lora-v5" \
    --base_path_to_store_results "/cluster/home/RL4LMs/results" \
    --entity_name youxiang \
    --log_to_wandb True \