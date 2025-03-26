#!/bin/bash

# Set environment variables
export HF_ENDPOINT="https://hf-mirror.com"
# export LIBRARY_PATH="/home/clouder/miniconda3/envs/llm/lib:$LIBRARY_PATH"

# Define the program and arguments
PROGRAM="/home/clouder/miniconda3/envs/llm/bin/deepspeed"
ARGS=(
    "--num_gpus" "1"  # 调试时建议单GPU
    "openrlhf/cli/train_ppo.py"
    #"--pretrain" "/home/clouder/LLaMA-Factory/saves/Qwen2.5-Coder-7B/PT/lora/train_2025-02-10-21-00-00"
    "--pretrain" "/home/clouder/OpenRLHF/checkpoint/Qwen2.5-Coder-7B_0307T23:16"
    "--remote_rm_url" "ose/ose_reward.py"
    "--save_path" "./checkpoint/Qwen2.5-Coder-7B"
    "--save_steps" "5"
    "--disable_ds_ckpt"
    "--ckpt_path" "./checkpoint/Qwen2.5-Coder-7B/ckpt"
    "--max_ckpt_num" "0"
    # "--load_checkpoint"
    "--logging_steps" "1"
    "--eval_steps" "-1"
    "--micro_train_batch_size" "2"
    "--train_batch_size" "64"
    "--micro_rollout_batch_size" "8"  # 注意，因为用了sglang或者vllm,这个跟一次推理多少个prompt没关系，只是跟后续计算kl,reward...的batch_size
    "--rollout_batch_size" "32"       # 取多少个prompt
    "--n_samples_per_prompt" "8"     # 每个prompt生成多个回复
    "--max_epochs" "1"                # 一个rollout生成的数据训练多少次
    "--num_episodes" "1"              # 总样本循环多少次
    # train step = rollout_batch_size * n_samples_per_prompt * max_epochs / train_batch_size
    # make experence = max_samples / rollout_batch_size
    "--prompt_max_len" "2048"
    "--generate_max_len" "2048"
    "--advantage_estimator" "reinforce"
    # "--use_kl_estimator_k3"
    # "--use_kl_loss"
    "--zero_stage" "1"
    "--bf16"
    "--lora_rank" "256"
    "--actor_learning_rate" "1e-6"    # "5e-5"
    "--critic_learning_rate" "9e-6"
    "--init_kl_coef" "0.00001"
    "--prompt_data" "ose/train_0shot_easy_medium_1epoch_free.json"  # "K-and-K/knights-and-knaves"
    "--input_key" "prompt"
    "--label_key" "tests"
    # "--apply_chat_template"
    "--max_samples" "10000"             # 10000
    "--temperature" "1.0"
    "--normalize_reward"
    "--adam_offload"                  # 启用adam_offload zero_stage就要选择1，不能选2，可能选2分散梯度的操作跟手动to_cpu不太兼容，具体原因待测试
    "--flash_attn"
    "--gradient_checkpointing"
    "--use_wandb" "7f53ddaa854c24f297d7d3185d16f0714c860fee"
    # "--actor_init_on_gpu"
)

# Run the program with the arguments
"$PROGRAM" "${ARGS[@]}"

# export HF_ENDPOINT="https://hf-mirror.com"
# HF_DATASETS_OFFLINE="1"

# python -m openrlhf.cli.lora_combiner \
#     --model_path Qwen/Qwen2.5-Coder-7B-Instruct \
#     --lora_path ./checkpoint/Qwen2.5-Coder-7B \
#     --output_path ./checkpoint/Qwen2.5-Coder-7B/export \
#     --bf16

# python /home/clouder/ose_code_model_data_preprocess/eval_test.py \
#     --model_path=./checkpoint/Qwen2.5-Coder-7B/export \
#     --output=./checkpoint/Qwen2.5-Coder-7B/ose_results

# accelerate launch \
#     /home/clouder/bigcode-evaluation-harness/main.py \
#     --model ./checkpoint/Qwen2.5-Coder-7B/export \
#     --max_length_generation 1024 \
#     --prompt codeqwen \
#     --eos "<|im_end|>" \
#     --tasks humanevalsynthesize-js,humanevalsynthesize-python \
#     --do_sample=False \
#     --n_samples 1 \
#     --batch_size 1 \
#     --allow_code_execution \
#     --precision bf16 \
#     --metric_output_path ./checkpoint/Qwen2.5-Coder-7B/code_results.json


# lm_eval --model hf \
#         --batch_size 8 \
#         --cache_requests true \
#         --model_args pretrained=./checkpoint/Qwen2.5-Coder-7B/export,dtype=bfloat16 \
#         --tasks mmlu,ceval-valid,ifeval \
#         --output_path ./checkpoint/Qwen2.5-Coder-7B/lm_results