# Large Scale MFA Training

Standalone MFA training code for large activation datasets.

You provide PyTorch `DataLoader`s. The repo handles initialization, training,
and checkpointing. `run_training.py` is the entry point to wrap in whatever job
system you use.

## Install

```bash
pip install -r requirements.txt
```

## Smoke Test

```bash
python3 run_training.py --config configs/hierarchy_demo_config.py
```

The smoke test uses `data/hierarchy_dataset_long.json` and converts text rows
into small hashed feature vectors.

## Real Run

1. Edit `configs/example_config.py`.
2. Implement `make_train_loader(layer)`.
3. Optionally implement `make_val_loader(layer)`.
4. Run `run_training.py`, or wrap it in your own job system.

Loader batches should be:

```python
x            # float tensor, shape (batch, activation_dim)
(x, tokens)  # optional token IDs for token-weighted K-Means
```

Direct run:

```bash
python3 run_training.py --config configs/example_config.py
```

Multi-GPU runs use the same entry point under your launcher, for example:

```bash
accelerate launch --num_processes 8 run_training.py --config configs/example_config.py
```

## Main Knobs

- `NUM_COMPONENTS`: pilot 1k-8k; serious run 8k-64k.
- `RANK`: start with 8 or 10.
- `KMEANS_POOL_SIZE`: 500k-2M for pilots, 2M-8M for larger dictionaries.
- `PROJECTED_DIM`: 256 is a good default.
- `STEPS_PER_EPOCH`: use a fixed step budget for huge streams.
