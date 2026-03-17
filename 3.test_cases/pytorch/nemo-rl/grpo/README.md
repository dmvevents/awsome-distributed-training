# Multi-node GRPO Training with NVIDIA NeMo RL on Amazon EKS (P5en)

## Overview

This is a test case for multi-node [GRPO](https://arxiv.org/abs/2402.03300) (Group Relative Policy Optimization) training using [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) on Amazon EKS with P5en instances (H200 GPUs). The setup uses [Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) as the base model and a synthetic math dataset for reinforcement learning from verifiable rewards.

NeMo RL orchestrates training with [Ray](https://docs.ray.io/) for cluster management, [vLLM](https://github.com/vllm-project/vllm) for fast generation rollouts, and PyTorch [DTensor](https://pytorch.org/docs/stable/distributed.tensor.html) for distributed policy optimization. All inter-node communication uses [EFA](https://aws.amazon.com/hpc/efa/) with NCCL RDMA for GPU-direct data transfer.

### Results

| Model | Accuracy (200 problems) |
|---|---|
| Nemotron-Mini-4B-Instruct (base) | 28% |
| After GRPO training (50 steps) | 35% (+7pp, +13 problems) |

## Architecture

```
+------------------------------------------------------------------+
|  Amazon EKS Cluster (2x P5en.48xlarge)                           |
|                                                                  |
|  +-----------------------------+  +----------------------------+ |
|  |  Head Node (8x H200)       |  | Worker Node (8x H200)      | |
|  |                             |  |                            | |
|  |  Ray Head + GCS             |  | Ray Worker                 | |
|  |  vLLM Generation Workers    |  | vLLM Generation Workers   | |
|  |  DTensor Policy Workers     |  | DTensor Policy Workers    | |
|  |  GRPO Orchestrator          |  |                            | |
|  +-------------|---------------+  +-------------|---------------+ |
|                |   EFA RDMA (16x 200Gbps = 3.2Tbps/node)  |     |
|                +-------------------------------------------+     |
|                                                                  |
|  FSx for Lustre (shared storage: model weights, checkpoints)    |
+------------------------------------------------------------------+
```

**Hardware per node:**
- 8x NVIDIA H200 (141GB HBM3e each) with NVSwitch
- 16x EFA devices (200Gbps each, 3.2Tbps aggregate per node)
- 192 vCPUs, 2TB RAM

**Software stack:**
- NeMo RL (GRPO algorithm with LoRA fine-tuning)
- Ray 2.49.2 (cluster orchestration, actor placement)
- vLLM (generation rollouts with tensor parallelism)
- PyTorch DTensor (distributed policy gradient with CPU offload)
- NCCL 2.27.5 + aws-ofi-nccl 1.18.0 + Libfabric 2.4 (EFA transport)

## Prerequisites

### EKS Cluster

You need an EKS cluster with at least 2x P5en.48xlarge nodes with EFA networking enabled. Follow the [Amazon EKS setup guide](https://github.com/aws-samples/awsome-distributed-training/tree/main/1.architectures/4.amazon-eks) or use [SageMaker HyperPod EKS](https://github.com/aws-samples/awsome-distributed-training/tree/main/1.architectures/7.sagemaker-hyperpod-eks).

Ensure the following are installed on your cluster:

1. **NVIDIA GPU Operator** (or NVIDIA device plugin):
   ```bash
   helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
   helm install gpu-operator nvidia/gpu-operator \
     --namespace gpu-operator --create-namespace \
     --set driver.enabled=false --set toolkit.enabled=false --wait
   ```

2. **AWS EFA Kubernetes Device Plugin**:
   ```bash
   helm repo add eks https://aws.github.io/eks-charts
   helm install efa eks/aws-efa-k8s-device-plugin -n kube-system
   ```

3. **Verify EFA devices** (P5en should show 16 per node):
   ```bash
   kubectl describe node <node-name> | grep vpc.amazonaws.com/efa
   ```

### Shared Storage

An FSx for Lustre filesystem mounted as a PersistentVolumeClaim. The training job uses shared storage for:
- Model weights and HuggingFace cache
- Training checkpoints (enables fault recovery)
- Training dataset

See [FSx for Lustre setup](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US/01-cluster/06-fsx-for-lustre) for instructions.

### Dataset

The training uses a synthetic math dataset with Python-verified answers. Generate it on a pod with FSx access:

```bash
pip install datasets
python generate_goldilocks_math.py --output /shared/nvrx-demo/goldilocks/train.jsonl --num-problems 10000
```

Or use any JSONL dataset with `problem` and `answer` fields.

## Docker Image

A pre-built image is available:

```
public.ecr.aws/v9l4g5s4/nemo-rl-workshop:p5en
```

To build from source, use the provided `Dockerfile`:

```bash
docker buildx build --platform linux/amd64 \
  -f Dockerfile \
  -t <your-ecr-repo>/nemo-rl:latest \
  --push .
```

The image includes:
- NeMo RL with vLLM and DTensor backends
- EFA networking stack (Libfabric 2.4, aws-ofi-nccl 1.18.0, GDRCopy)
- H200/NVSwitch NCCL workarounds (`NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`)
- Pre-fetched Ray worker virtual environments

See `sbom/` for the full Software Bill of Materials and third-party licenses.

## Launching GRPO Training

### Quick Start

1. **Edit the manifest** to match your cluster (namespace, PVC name, node selector):
   ```bash
   vim rayjob-grpo.yaml
   ```

2. **Deploy the RayJob**:
   ```bash
   kubectl apply -f rayjob-grpo.yaml
   ```

3. **Monitor training**:
   ```bash
   # Watch pod status
   kubectl get pods -n <namespace> -l app=grpo-nvrx-rayjob -w

   # Stream training logs from the head pod
   HEAD=$(kubectl get pod -n <namespace> -l app=grpo-nvrx-rayjob,ray.io/node-type=head \
     -o jsonpath='{.items[0].metadata.name}')
   kubectl logs -f $HEAD -n <namespace>
   ```

### Training Configuration

The entrypoint script (`rayjob_entrypoint.sh`) runs GRPO with these key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `cluster.num_nodes` | 2 | Number of P5en nodes |
| `cluster.gpus_per_node` | 8 | H200 GPUs per node |
| `grpo.max_num_steps` | 50 | Training steps |
| `grpo.num_prompts_per_step` | 16 | Prompts per GRPO step |
| `grpo.num_generations_per_prompt` | 16 | Rollouts per prompt |
| `policy.dtensor_cfg.lora_cfg.enabled` | true | LoRA fine-tuning |
| `policy.dtensor_cfg.lora_cfg.dim` | 32 | LoRA rank |
| `policy.dtensor_cfg.lora_cfg.alpha` | 64 | LoRA scaling |
| `policy.dtensor_cfg.lora_cfg.target_modules` | q,k,v,o_proj | Attention-only LoRA |
| `policy.optimizer.kwargs.lr` | 5e-5 | Learning rate |
| `loss_fn.reference_policy_kl_penalty` | 0.04 | KL divergence penalty |

### NCCL Environment Variables

The following NCCL settings are required for P5en (H200 with NVSwitch):

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_CUMEM_ENABLE` | 0 | Workaround for cuMem import penalty ([nccl#1749](https://github.com/NVIDIA/nccl/issues/1749)) |
| `NCCL_NVLS_ENABLE` | 0 | Workaround for NVLS rank ordering bug ([nccl#1906](https://github.com/NVIDIA/nccl/issues/1906)) |
| `FI_PROVIDER` | efa | Use EFA libfabric provider |
| `FI_EFA_USE_DEVICE_RDMA` | 1 | Enable GPU Direct RDMA |
| `NCCL_NET_PLUGIN` | ofi | Use aws-ofi-nccl plugin |
| `NCCL_SOCKET_IFNAME` | ^lo,docker,veth,eni | Exclude non-data interfaces |

These are baked into the container image but also set explicitly in the RayJob manifest.

## Evaluation

After training completes, evaluate the model against the held-out test set:

```bash
# On a pod with GPU access and the FSx mount
python eval_nemotron_goldilocks.py \
  --model nvidia/Nemotron-Mini-4B-Instruct \
  --checkpoint-dir /shared/nvrx-demo/phase2-checkpoints
```

The script evaluates both the base model and the LoRA-trained checkpoint on 200 held-out problems, reporting accuracy and format compliance (`\boxed{}` usage).

Sample output:
```
  BEFORE vs AFTER COMPARISON
  Accuracy:  57/200 (28%)  ->  70/200 (35%)
  \boxed{} : 12/200 (6%)  ->  180/200 (90%)

  Accuracy improved by 13 problems (+7pp)
  Format compliance improved by 168 (84pp)
```

## Fault Tolerance

The RayJob manifest includes `backoffLimit: 2`, which means Kubernetes will retry the job up to 2 times on failure. Combined with checkpoint saving to FSx every 25 steps, the training automatically resumes from the latest checkpoint after a fault.

To test fault injection:

```bash
# Find the head pod
HEAD=$(kubectl get pod -n <namespace> -l app=grpo-nvrx-rayjob,ray.io/node-type=head \
  -o jsonpath='{.items[0].metadata.name}')

# Kill the training process
kubectl exec $HEAD -n <namespace> -- bash -c \
  'PID=$(ps aux | grep "[r]un_grpo" | head -1 | awk "{print \$2}"); kill -9 $PID'

# Watch the RayJob restart and resume from checkpoint
kubectl get rayjob -n <namespace> -w
```

## Known Issues

| Issue | Impact | Workaround |
|-------|--------|------------|
| [NVIDIA/nccl#2024](https://github.com/NVIDIA/nccl/issues/2024) | Ring Broadcast hangs >16M elements on H200/NVSwitch with Ray | Use `NCCL_ALGO=Tree` or broadcast-via-AllReduce patch |
| [ray-project/ray#61073](https://github.com/ray-project/ray/issues/61073) | NCCL AllReduce 2400x slower via Ray on H200 | Set `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` |
| [NVIDIA/nccl#1749](https://github.com/NVIDIA/nccl/issues/1749) | cuMem import penalty causes slow first collective | Set `NCCL_CUMEM_ENABLE=0` |
| [NVIDIA/nccl#1906](https://github.com/NVIDIA/nccl/issues/1906) | NVLS rank ordering mismatch | Set `NCCL_NVLS_ENABLE=0` |

## Upstream Contributions

This work resulted in the following upstream contributions:

- [NVIDIA/nvidia-resiliency-ext#278](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/278) - NVRx fault tolerance integration
- [ai-dynamo/dynamo#7390](https://github.com/ai-dynamo/dynamo/pull/7390) - Local model path support for Dynamo inference

## File Structure

```
grpo/
  README.md                       This file
  Dockerfile                      Container build (NeMo RL + EFA stack)
  rayjob-grpo.yaml                Kubernetes RayJob manifest (2x P5en)
  rayjob_entrypoint.sh            Training entrypoint script
  eval_nemotron_goldilocks.py     Before/after evaluation script
  sbom/
    sbom.txt                      Software Bill of Materials
    pip-freeze.txt                Python package versions
    dpkg.txt                      System package versions
```

## References

- [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) - GRPO training framework
- [Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) - Base model
- [GRPO Paper](https://arxiv.org/abs/2402.03300) - DeepSeekMath: Integrating RL with LLMs
- [Ray Documentation](https://docs.ray.io/) - Distributed computing framework
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput inference engine
- [AWS EFA Developer Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) - Elastic Fabric Adapter
- [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) - NCCL OFI plugin for EFA
