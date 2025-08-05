export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_SOCKET_IFNAME=^lo,docker0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
  --model Qwen/Qwen3-0.6B \
  --train_type full \
  --dataset '../../datasets_example/TFRank_sft_training_data/msmarco/train' \
            '/datasets_example/TFRank_sft_training_data/msmarco-fg/no_think/train' \
            '/datasets_example/TFRank_sft_training_data/msmarco-fg/think/train' \
  --val_dataset '/datasets_example/TFRank_sft_training_data/msmarco/valid' \
            '/datasets_example/TFRank_sft_training_data/msmarco-fg/no_think/valid' \
            '/datasets_example/TFRank_sft_training_data/msmarco-fg/think/valid' \
  --torch_dtype bfloat16 \
  --split_dataset_ratio 0 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --save_strategy 'steps' \
  --eval_strategy 'steps' \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --output_dir output/tfrank_sft-qwen3-0.6B \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 128 \
  --deepspeed zero2 \
  --report_to tensorboard \
  --use_liger_kernel true \
  --truncation_strategy delete \
  --packing true \
  --sequence_parallel_size 4 \
  --padding_free true \
  --attn_impl flash_attn \
  > output/tfrank_sft-qwen3-0.6B/train.log 2>&1