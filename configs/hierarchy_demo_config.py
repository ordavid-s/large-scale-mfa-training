from pathlib import Path

import torch

from demo_data.hierarchy_json import make_hierarchy_dataloaders


REPO_ROOT = Path(__file__).resolve().parents[1]
HIERARCHY_JSON_PATH = REPO_ROOT / "data" / "hierarchy_dataset_long.json"

# Outputs.
OUTPUT_DIR = Path("runs/hierarchy_demo")
RUN_NAME = "hierarchy_demo_mfa"

# What to train.
MODEL_NAME = "hierarchy-hashed-demo"
LAYERS = [0]
NUM_COMPONENTS = 16
RANK = 4

# Runtime. This demo is deliberately CPU-friendly.
TRAINING_BACKEND = "single_gpu"
DEVICE = "cpu"
DATA_DTYPE = torch.float32
SEED = 0

# Demo DataLoader setup.
FEATURE_DIM = 128
BATCH_SIZE = 256
NUM_WORKERS = 0
PIN_MEMORY = False
VAL_FRACTION = 0.2
MAX_EXAMPLES = 2_000


def make_loaders(layer):
    train_loader, val_loader, _level_to_id = make_hierarchy_dataloaders(
        HIERARCHY_JSON_PATH,
        feature_dim=FEATURE_DIM,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        max_examples=MAX_EXAMPLES,
        seed=SEED + int(layer),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        dtype=DATA_DTYPE,
    )
    return train_loader, val_loader


# Initialization. Use "random" or "projected_kmeans".
INIT_METHOD = "projected_kmeans"
CENTROIDS_PATH = OUTPUT_DIR / "centroids" / "centroids_{model_name}_L{layer}_k{num_components}.pt"
FORCE_REINIT = False

# Small projected K-Means settings for a quick smoke test.
KMEANS_POOL_SIZE = 512
PROJECTED_DIM = 32
MODEL_VOCAB_SIZE = 16
USE_TOKEN_WEIGHTS = False
KMEANS_SMOOTHING = 1.0
KMEANS_POWER = 1.0
KMEANS_ITERS = 10
KMEANS_RESTARTS = 2
KMEANS_TOL = 1e-4
KMEANS_METRIC = "euclidean"
KMEANS_REFINE_EPOCHS = 2

# MFA model initialization.
PSI_INIT = 1.0
PSI_PER_COMPONENT = False
SCALE_INIT = 1.0
EPS_FLOOR = 1e-5

# Optimization.
NUM_EPOCHS = 2
LR = 1e-3
GRAD_CLIP = None
LOG_INTERVAL = 5
STEPS_PER_EPOCH = None
START_EPOCH = 1

# MGPU-only logging/checkpoint cadence.
VAL_INTERVAL = 100
WANDB_PROJECT = None

# Resume from an explicit checkpoint, or leave None to auto-use the layer output path.
RESUME_CHECKPOINT = None
