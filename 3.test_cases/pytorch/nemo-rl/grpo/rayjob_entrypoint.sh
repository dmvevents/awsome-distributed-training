#!/bin/bash
set -e
echo "=============================================="
echo "PHASE 2: Accuracy Sprint (merged base + protective LoRA)"
echo "Starting from step_60 merged model (has format knowledge)"
echo "=============================================="
MY_IP=$(hostname -I | awk '{print $1}')
echo "$MY_IP $(hostname)" >> /etc/hosts
export NCCL_NET_PLUGIN=ofi
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}
GPU_COUNT=$(nvidia-smi -L | wc -l)
CKPT_DIR="/shared/nvrx-demo/phase2-checkpoints"
# Only clear checkpoints on first run — preserve for fault recovery resume
if [ "${CLEAR_CHECKPOINTS:-0}" = "1" ]; then
  echo "Clearing checkpoints (CLEAR_CHECKPOINTS=1)"
  rm -rf "$CKPT_DIR"/*
fi
mkdir -p "$CKPT_DIR"
EXISTING=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | wc -l)
[ "$EXISTING" -gt 0 ] && echo "Found $EXISTING existing checkpoint(s) — will resume from latest"
export HF_HOME="/shared/nvrx-demo/hf_cache"
export HF_DATASETS_CACHE="/shared/nvrx-demo/hf_datasets"
export TORCH_HOME="/shared/nvrx-demo/torch_cache"
export TRITON_CACHE_DIR="/shared/nvrx-demo/triton_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"

# Step 1: Merge phase 1 LoRA if not already done
MERGED="/shared/nvrx-demo/nemotron-phase1-merged"
if [ ! -f "$MERGED/model.safetensors" ]; then
  echo "Merging phase 1 LoRA checkpoint..."
  pip install peft -q 2>&1 | tail -1
  python3 /shared/nvrx-demo/scripts/merge_lora_checkpoint.py \
    --model nvidia/Nemotron-Mini-4B-Instruct \
    --checkpoint-dir /shared/nvrx-demo/miracle-checkpoints \
    --output "$MERGED"
else
  echo "Phase 1 merged model already exists at $MERGED"
fi

cd /opt/nemo-rl

echo "  Phase 2: attention-only LoRA on merged model"
echo "  Gemini config: epochs=2, KL=0.04, lr=5e-5, alpha=64(scale=2.0)"
echo "  50 steps (~15 min)"

exec python3 examples/run_grpo.py \
    cluster.num_nodes=2 \
    cluster.gpus_per_node=$GPU_COUNT \
    grpo.max_num_steps=${GRPO_MAX_STEPS:-50} \
    grpo.max_num_epochs=2 \
    grpo.num_prompts_per_step=16 \
    grpo.num_generations_per_prompt=16 \
    grpo.use_dynamic_sampling=true \
    grpo.batch_multiplier=4 \
    grpo.dynamic_sampling_max_gen_batches=10 \
    grpo.overlong_filtering=true \
    policy.model_name="$MERGED" \
    policy.tokenizer.name=nvidia/Nemotron-Mini-4B-Instruct \
    policy.megatron_cfg.enabled=false \
    policy.dtensor_cfg.enabled=true \
    policy.dtensor_cfg._v2=true \
    policy.dtensor_cfg.tensor_parallel_size=1 \
    policy.dtensor_cfg.activation_checkpointing=true \
    policy.dtensor_cfg.cpu_offload=true \
    policy.dtensor_cfg.lora_cfg.enabled=true \
    policy.dtensor_cfg.lora_cfg.dim=32 \
    policy.dtensor_cfg.lora_cfg.alpha=64 \
    policy.dtensor_cfg.lora_cfg.dropout=0.0 \
    policy.dtensor_cfg.lora_cfg.dropout_position=pre \
    policy.dtensor_cfg.lora_cfg.lora_A_init=xavier \
    'policy.dtensor_cfg.lora_cfg.target_modules=[q_proj,k_proj,v_proj,o_proj]' \
    'policy.dtensor_cfg.lora_cfg.exclude_modules=[]' \
    policy.optimizer.kwargs.lr=5e-5 \
    policy.train_global_batch_size=256 \
    policy.train_micro_batch_size=16 \
    policy.max_total_sequence_length=1024 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.4 \
    ++loss_fn.reference_policy_kl_penalty=0.04 \
    ++data.train.dataset_name=ResponseDataset \
    ++data.train.data_path=/shared/nvrx-demo/goldilocks/train.jsonl \
    ++data.train.input_key=problem \
    ++data.train.output_key=answer \
    ++data.train.split_validation_size=0.05 \
    logger.wandb_enabled=false \
    logger.log_dir=/tmp/grpo-logs \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    checkpointing.save_period=25 \
    checkpointing.keep_top_k=5
