# Large Scale MFA Training

Standalone MFA training code for large activation datasets.

The code just requires a PyTorch `DataLoader`. The repo has code for initialization, training,
and checkpointing. `run_training.py` is the entry point for running training, but `notebooks/train_large_scale_mfa.ipynb` has a demo that explains the different hyperparameters and how to run the pipeline from end to end. Best to start there.

## Install

```bash
pip install -r requirements.txt
```

## Real Run

1. Edit `configs/example_config.py`.
2. Implement `make_train_loader(layer)`.
3. Optionally implement `make_val_loader(layer)`.
4. Run `run_training.py`, or wrap it in your own job system.

Use `RUN_NAME` for the filesystem-safe experiment name. Use `MODEL_NAME` for the
source model that produced the activations, for example `meta-llama/Llama-3.1-8B`.
Only used for saving and loading purposes.

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
