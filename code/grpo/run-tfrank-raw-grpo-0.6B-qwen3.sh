export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_SOCKET_IFNAME=^lo,docker0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen3-0.6B \
  --external_plugins ./rank_plugin.py \
  --reward_funcs think_rank_reward think_format_reward \
  --reward_weights 1.0 0.5 \
  --vllm_mode server \
  --use_vllm true \
  --train_type full \
  --dataset '../../datasets_example/TFRank_grpo_training_data/msmarco-fg/no_think/train' \
            '../../datasets_example/TFRank_grpo_training_data/msmarco-fg/think/train' \
            '../../datasets_example/TFRank_grpo_training_data/msmarco/no_think/train' \
  --val_dataset '../../datasets_example/TFRank_grpo_training_data/msmarco-fg/no_think/valid' \
            '../../datasets_example/TFRank_grpo_training_data/msmarco-fg/think/valid' \
            '../../datasets_example/TFRank_grpo_training_data/msmarco/no_think/valid' \
  --torch_dtype bfloat16 \
  --split_dataset_ratio 0. \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --save_strategy 'steps' \
  --eval_strategy 'steps' \
  --eval_steps 500 \
  --save_steps 500 \
  --max_completion_length 4096 \
  --overlong_filter True \
  --vllm_gpu_memory_utilization 0.5 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --output_dir output/tfrank_raw_grpo-qwen3-0.6B \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 128 \
  --num_generations 4 \
  --temperature 0.9 \
  --log_completions true \
  --async_generate False \
  --deepspeed zero2 \
  --report_to tensorboard \
  --attn_impl flash_attn \
  > output/tfrank_raw_grpo-qwen3-0.6B/train.log 2>&1