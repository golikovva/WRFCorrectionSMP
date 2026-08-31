# DDP training

Distributed mode is enabled automatically when the program is launched with
`torchrun`. A regular Python launch keeps the existing single-process behavior.

## One GPU

```bash
python experiments/train_test/main.py --cfg configs/train_test.yaml
```

## Two GPUs on one cluster node

```bash
torchrun --standalone --nproc_per_node=2 \
  experiments/train_test/main.py --cfg configs/train_test.yaml
```

For the multi-stage pipeline:

```bash
torchrun --standalone --nproc_per_node=2 \
  experiments/multi_domain/main.py configs/multi_domain.yaml
```

`run_config.batch_size` is the batch size **per GPU**. Thus, two processes use
an effective global batch size of `2 * run_config.batch_size`. Reduce the
configured value if the old value should remain the global batch size.

Every rank participates in training and validation and receives a disjoint
shard of dates. Validation loss and component statistics are summed across
ranks; only rank 0 logs the combined result and writes checkpoints. Testing is
still performed only by rank 0. During a long test, the other ranks release
their model-related CUDA memory and poll a completion file from the CPU instead
of waiting in an NCCL collective. This avoids NCCL timeouts and lets the ranks
rejoin a following multi-domain stage.
The sampler pads the final shard when the number of dates is not divisible by
the number of GPUs, so all ranks execute the same number of optimization steps.

The included `run_wrf_corr.sbatch` requests two GPUs and starts two processes.
For a one-GPU job, request `gpu:1` and use `python main.py` (or use
`torchrun --nproc_per_node=1`, which also remains non-distributed).
