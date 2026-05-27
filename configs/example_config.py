"""
Large-scale MFA training template.

Provide PyTorch DataLoader factories below. The loader can read from memmap,
WebDataset, HDF5, Parquet, zarr, a database, object storage, or any internal
format.

Batch contract:
  x
  (x, tokens_or_metadata)

x must be a float tensor of shape (batch, activation_dim). Tokens are only used
when USE_TOKEN_WEIGHTS=True for projected K-Means initialization.
"""

from pathlib import Path

import torch


# Run identity.
OUTPUT_DIR = Path("/path/to/output/mfa_runs")
RUN_NAME = "my_large_scale_mfa"
MODEL_NAME = "my_model"
LAYERS = [0]

# Model size.
NUM_COMPONENTS = 8_000
RANK = 10

# Runtime.
TRAINING_BACKEND = "mgpu"  # "single_gpu" or "mgpu"
DEVICE = "cuda"
DATA_DTYPE = torch.float32
SEED = 0

# Loader settings for your implementation.
BATCH_SIZE = 256
NUM_WORKERS = 8
PIN_MEMORY = True


def make_train_loader(layer):
    """
    Replace with your activation DataLoader.

    For 100B-scale runs, use a streaming Dataset and set STEPS_PER_EPOCH below.
    """
    raise NotImplementedError("Define make_train_loader(layer) for your activation store.")


def make_val_loader(layer):
    """
    Return a small fixed validation loader, or None.
    """
    return None


# Initialization.
INIT_METHOD = "projected_kmeans"
CENTROIDS_PATH = OUTPUT_DIR / "centroids" / "centroids_{model_name}_L{layer}_k{num_components}.pt"
FORCE_REINIT = False

KMEANS_POOL_SIZE = 4_000_000
PROJECTED_DIM = 256
KMEANS_ITERS = 50
KMEANS_RESTARTS = 5
KMEANS_REFINE_EPOCHS = 3
KMEANS_TOL = 1e-4
KMEANS_METRIC = "euclidean"

USE_TOKEN_WEIGHTS = False
MODEL_VOCAB_SIZE = 256_000
KMEANS_SMOOTHING = 1.0
KMEANS_POWER = 1.0

# MFA initialization.
PSI_INIT = 1.0
PSI_PER_COMPONENT = False
SCALE_INIT = 1.0
EPS_FLOOR = 1e-5

# Optimization.
NUM_EPOCHS = 5
STEPS_PER_EPOCH = 50_000
LR = 7e-5
GRAD_CLIP = None
LOG_INTERVAL = 100
START_EPOCH = 1

# MGPU logging/checkpoint cadence.
VAL_INTERVAL = 10_000
WANDB_PROJECT = None
RESUME_CHECKPOINT = None


# Quick scale guide:
# NUM_COMPONENTS: pilot 1k-8k; serious 8k-64k.
# RANK: start with 8 or 10.
# KMEANS_POOL_SIZE: 500k-2M for pilots, 2M-8M for larger dictionaries.
# STEPS_PER_EPOCH: use a fixed step budget for huge streams.
