PORT=8113
MODEL_NAME='rele_pointwise'
MODEL_PATH='Johnnyfans/TFRank-GRPO-Qwen3-0.6B'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --task auto \
    --port $PORT \
    --tensor-parallel-size 4 \