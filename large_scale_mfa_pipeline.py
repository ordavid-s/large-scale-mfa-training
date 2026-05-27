from __future__ import annotations

import random
from inspect import signature
from pathlib import Path
from typing import Any


def _cfg(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default)


def _torch_dtype(value: Any):
    import torch

    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.replace("torch.", "")
        if hasattr(torch, normalized):
            dtype = getattr(torch, normalized)
            if isinstance(dtype, torch.dtype):
                return dtype
    raise ValueError(f"Unsupported DATA_DTYPE: {value!r}")


def _format_path(template: Any, cfg: Any, layer: int) -> Path | None:
    if template is None:
        return None
    text = str(template).format(
        model_name=_cfg(cfg, "MODEL_NAME", "model"),
        run_name=_cfg(cfg, "RUN_NAME", "run"),
        layer=layer,
        num_components=int(_cfg(cfg, "NUM_COMPONENTS")),
        rank=int(_cfg(cfg, "RANK")),
    )
    return Path(text).expanduser()


def _layer_save_path(cfg: Any, layer: int) -> Path:
    output_dir = Path(_cfg(cfg, "OUTPUT_DIR", "runs/default")).expanduser()
    run_name = _cfg(cfg, "RUN_NAME", None)
    if run_name is None:
        run_name = f"mfa_k{_cfg(cfg, 'NUM_COMPONENTS')}_r{_cfg(cfg, 'RANK')}"
    return output_dir / "models" / f"{run_name}_L{layer}.ckpt"


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _call_loader_factory(factory: Any, layer: int):
    if not callable(factory):
        return factory

    params = signature(factory).parameters
    if "layer" in params:
        return factory(layer=layer)

    positional = [
        p for p in params.values()
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if positional:
        return factory(layer)

    return factory()


def make_loaders(cfg: Any, layer: int):
    """
    Return PyTorch loaders from the config.

    Supported config styles:
      - make_loaders(layer) -> (train_loader, val_loader)
      - make_train_loader(layer), optional make_val_loader(layer)
      - TRAIN_LOADER, optional VAL_LOADER

    Batches may be either x or (x, token_or_metadata).
    """
    if hasattr(cfg, "make_loaders"):
        loaders = _call_loader_factory(cfg.make_loaders, layer)
        if not isinstance(loaders, (tuple, list)) or len(loaders) != 2:
            raise ValueError("make_loaders(layer) must return (train_loader, val_loader).")
        return loaders[0], loaders[1]

    train_factory = _cfg(cfg, "make_train_loader", None) or _cfg(cfg, "TRAIN_LOADER", None)
    if train_factory is None:
        raise ValueError("Config must define make_train_loader(layer), make_loaders(layer), or TRAIN_LOADER.")

    val_factory = _cfg(cfg, "make_val_loader", None) or _cfg(cfg, "VAL_LOADER", None)

    train_loader = _call_loader_factory(train_factory, layer)
    val_loader = _call_loader_factory(val_factory, layer) if val_factory is not None else None
    return train_loader, val_loader


def initialize_centroids(cfg: Any, layer: int, train_loader):
    import torch

    seed = int(_cfg(cfg, "SEED", 0))
    k = int(_cfg(cfg, "NUM_COMPONENTS"))
    method = str(_cfg(cfg, "INIT_METHOD", "projected_kmeans")).lower()
    centroids_path = _format_path(_cfg(cfg, "CENTROIDS_PATH", None), cfg, layer)

    if centroids_path is not None and centroids_path.exists() and not bool(_cfg(cfg, "FORCE_REINIT", False)):
        print(f"[init] loading centroids from {centroids_path}")
        return torch.load(centroids_path, map_location="cpu")

    if method == "random":
        from initializations.random_init import sample_centroids_stream_uniform_with_replacement

        print(f"[init] sampling {k} random centroids")
        centroids = sample_centroids_stream_uniform_with_replacement(
            train_loader,
            K=k,
            seed=seed,
            out_device=torch.device(_cfg(cfg, "DEVICE", "cuda")),
            out_dtype=_torch_dtype(_cfg(cfg, "DATA_DTYPE", torch.float32)),
        )
    elif method == "projected_kmeans":
        from initializations.projected_knn import ReservoirKMeans

        print(f"[init] running projected K-Means for {k} centroids")
        knn = ReservoirKMeans(
            n_clusters=k,
            pool_size=int(_cfg(cfg, "KMEANS_POOL_SIZE")),
            vocab_size=_cfg(cfg, "MODEL_VOCAB_SIZE", None),
            smoothing=float(_cfg(cfg, "KMEANS_SMOOTHING", 1.0)),
            power=float(_cfg(cfg, "KMEANS_POWER", 1.0)),
            kmeans_iters=int(_cfg(cfg, "KMEANS_ITERS", 50)),
            kmeans_restarts=int(_cfg(cfg, "KMEANS_RESTARTS", 10)),
            tol=float(_cfg(cfg, "KMEANS_TOL", 1e-4)),
            seed=seed,
            device=torch.device(_cfg(cfg, "DEVICE", "cuda")),
            metric=str(_cfg(cfg, "KMEANS_METRIC", "euclidean")),
            proj_dim=_cfg(cfg, "PROJECTED_DIM", 256),
        )
        token_loader = train_loader if bool(_cfg(cfg, "USE_TOKEN_WEIGHTS", False)) else None
        centroids = knn.fit(
            train_loader,
            token_loader=token_loader,
            refine_epochs=int(_cfg(cfg, "KMEANS_REFINE_EPOCHS", 5)),
        )
    else:
        raise ValueError("INIT_METHOD must be 'random' or 'projected_kmeans'.")

    if centroids_path is not None:
        centroids_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(centroids.detach().cpu(), centroids_path)
        print(f"[init] saved centroids to {centroids_path}")

    return centroids


def build_or_load_model(cfg: Any, layer: int, train_loader):
    import torch
    from modeling.mfa import MFA

    backend = str(_cfg(cfg, "TRAINING_BACKEND", "single_gpu")).lower()
    save_path = _layer_save_path(cfg, layer)
    resume = _format_path(_cfg(cfg, "RESUME_CHECKPOINT", None), cfg, layer)
    resume_path = resume or (save_path if save_path.exists() else None)

    if resume_path is not None and Path(resume_path).exists():
        print(f"[model] loading checkpoint from {resume_path}")
        if backend == "mgpu":
            from modeling.model_checkpointing import load_mfa
        else:
            from modeling.mfa import load_mfa
        return load_mfa(str(resume_path))

    centroids = initialize_centroids(cfg, layer, train_loader)
    model = MFA(
        centroids,
        rank=int(_cfg(cfg, "RANK")),
        psi_init=float(_cfg(cfg, "PSI_INIT", 1.0)),
        psi_per_component=bool(_cfg(cfg, "PSI_PER_COMPONENT", False)),
        scale_init=float(_cfg(cfg, "SCALE_INIT", 1.0)),
        eps_floor=float(_cfg(cfg, "EPS_FLOOR", 1e-5)),
    )
    if backend == "single_gpu":
        model = model.to(torch.device(_cfg(cfg, "DEVICE", "cuda")))
    return model


def train_layer(cfg: Any, layer: int) -> dict[str, Any]:
    import torch

    backend = str(_cfg(cfg, "TRAINING_BACKEND", "single_gpu")).lower()
    save_path = _layer_save_path(cfg, layer)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[layer {layer}] preparing loaders")
    train_loader, val_loader = make_loaders(cfg, layer)
    model = build_or_load_model(cfg, layer, train_loader)

    print(f"[layer {layer}] training with backend={backend}")
    if backend == "single_gpu":
        from modeling.mfa import save_mfa
        from modeling.train import train_nll

        result = train_nll(
            model,
            train_loader,
            val_loader=val_loader,
            epochs=int(_cfg(cfg, "NUM_EPOCHS", 5)),
            lr=float(_cfg(cfg, "LR", 1e-3)),
            grad_clip=_cfg(cfg, "GRAD_CLIP", None),
            save_path=str(save_path),
            save_func=save_mfa,
            log_interval=int(_cfg(cfg, "LOG_INTERVAL", 100)),
            steps_per_epoch=_cfg(cfg, "STEPS_PER_EPOCH", None),
        )
        save_mfa(model, str(save_path))
    elif backend == "mgpu":
        from modeling.model_checkpointing import save_mfa
        from modeling.train_mgpu import train_nll

        result = train_nll(
            model,
            loader=train_loader,
            val_loader=val_loader,
            epochs=int(_cfg(cfg, "NUM_EPOCHS", 5)),
            lr=float(_cfg(cfg, "LR", 1e-3)),
            grad_clip=_cfg(cfg, "GRAD_CLIP", None),
            save_path=str(save_path),
            save_func=save_mfa,
            log_interval=int(_cfg(cfg, "LOG_INTERVAL", 100)),
            val_interval=int(_cfg(cfg, "VAL_INTERVAL", 100_000)),
            steps_per_epoch=_cfg(cfg, "STEPS_PER_EPOCH", None),
            wandb_project=_cfg(cfg, "WANDB_PROJECT", None),
            run_name=f"{_cfg(cfg, 'RUN_NAME', 'mfa')}_L{layer}",
            start_epoch=int(_cfg(cfg, "START_EPOCH", 1)),
        )
    else:
        raise ValueError("TRAINING_BACKEND must be 'single_gpu' or 'mgpu'.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[layer {layer}] done. model path: {save_path}")
    return {"layer": layer, "save_path": str(save_path), "result": result}


def run_from_config(cfg: Any) -> list[dict[str, Any]]:
    _seed_everything(int(_cfg(cfg, "SEED", 0)))
    layers = list(_cfg(cfg, "LAYERS", [0]))
    if not layers:
        raise ValueError("LAYERS must contain at least one layer.")
    results = []
    for layer in layers:
        results.append(train_layer(cfg, int(layer)))
    return results
