#!/bin/bash
PORT=8113
VLLM_MODEL_NAME='rele_pointwise'
MODEL_NAME='checkpoint'

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_SOCKET_IFNAME=^lo,docker0

# 启动 vllm 服务到后台
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --served-model-name $VLLM_MODEL_NAME \
    --task auto \
    --port $PORT \
    --tensor-parallel-size 4 \
    > vllm.log 2>&1 &

VLLM_PID=$!

# 等待 vllm 服务端口就绪
for i in {1..60}; do
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null; then
        echo "vLLM 服务已启动"
        break
    fi
    sleep 2
done

API_KEY="any-string"
API_BASE="http://localhost:$PORT/v1"
INPUT_DIR="./input/BRIGHT"
RESULT_DIR="./rank_results"
MAX_NEW_TOKENS_INIT=128
THINK_MODE=false 
REASONING_MODEL=true 
TEMPERATURE=0.0
DEBUG=false

python run_eval.py \
  --model_name "$MODEL_NAME" \
  --api_key "$API_KEY" \
  --api_base "$API_BASE" \
  --input_dir "$INPUT_DIR" \
  --result_dir "$RESULT_DIR" \
  --max_new_tokens_init $MAX_NEW_TOKENS_INIT \
  --think_mode $THINK_MODE \
  --reasoning_model $REASONING_MODEL \
  --temperature $TEMPERATURE \
  --debug $DEBUG 

# 推理结束后关闭 vllm 服务
kill $VLLM_PID