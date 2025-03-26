#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

deepspeed --num_gpus 1 \
    openrlhf/cli/train_ppo.py \
    --pretrain "Qwen/Qwen2.5-7B-Instruct" \
    --remote_rm_url "kk/kk.py" \
    --save_path "./checkpoint/Qwen2.5-7B-5ppl" \
    --save_steps 5 \
    --disable_ds_ckpt \
    --ckpt_path "./checkpoint/Qwen2.5-7B-5ppl/ckpt" \
    --max_ckpt_num 0 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 4 \
    --train_batch_size 8 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 16 \
    --n_samples_per_prompt 16 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 2048 \
    --advantage_estimator reinforce_baseline \
    --zero_stage 1 \
    --bf16 \
    --lora_rank 256 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.001 \
    --prompt_data "kk/3ppl/train.parquet,kk/4ppl/train.parquet,kk/5ppl/train.parquet" \
    --input_key "prompt" \
    --label_key "ground_truth" \
    --max_samples 10000 \
    --num_episodes 1 \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --use_wandb "7f53ddaa854c24f297d7d3185d16f0714c860fee"
    # --use_kl_loss \
    # --use_kl_estimator_k3
    

# python -m openrlhf.cli.lora_combiner \
#     --model_path Qwen/Qwen2.5-7B-Instruct \
#     --lora_path ./checkpoint/Qwen2.5-7B-3ppl \
#     --output_path ./checkpoint/Qwen2.5-7B-3ppl/export \
#     --bf16