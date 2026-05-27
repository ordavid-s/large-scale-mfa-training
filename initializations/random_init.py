import torch
from typing import Iterable, Optional


def _acts_from_batch(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


@torch.no_grad()
def sample_centroids_stream_uniform_with_replacement(
    loader: Iterable,
    *,
    K: int,
    seed: Optional[int] = 0,
    out_device: Optional[torch.device] = None,  # where to store centroids
    out_dtype: Optional[torch.dtype] = None,    # dtype for centroids
) -> torch.Tensor:
    """
    EXACT streaming sampling of K items uniformly WITH replacement from the full stream.

    Equivalent to: after seeing all N items, return K i.i.d. samples from {1..N}
    (uniform), allowing duplicates — without knowing N in advance and without a pool.

    Loader batches may be x or (x, ...), where x has shape (B, D).

    Returns:
      centroids: (K,D)
    """
    if K < 1:
        raise ValueError("K must be >= 1")

    g = None
    if seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

    centroids = None
    seen = 0  # total items seen so far (n)

    for batch in loader:
        acts = _acts_from_batch(batch)
        if acts.ndim != 2:
            raise ValueError(f"expected acts to be (B,D), got {tuple(acts.shape)}")
        B, D = acts.shape

        if centroids is None:
            dev = acts.device if out_device is None else out_device
            dt = acts.dtype if out_dtype is None else out_dtype
            centroids = torch.empty((K, D), device=dev, dtype=dt)

        # Process items in the batch sequentially (prob depends on global index n)
        for i in range(B):
            seen += 1
            x = acts[i]

            # For each slot k, replace with probability 1/seen
            # Vectorized across K: draw K uniforms and compare to 1/seen
            u = torch.rand((K,), generator=g)  # CPU
            mask = (u < (1.0 / seen))          # (K,) bool on CPU

            if mask.any():
                # Move mask to centroid device and write those rows
                m = mask.to(device=centroids.device)
                centroids[m] = x.to(device=centroids.device, dtype=centroids.dtype)

    if centroids is None:
        raise ValueError("loader yielded no data")
    if seen == 0:
        raise ValueError("no items seen")

    return centroids


# Example:
# centroids = sample_centroids_stream_uniform_with_replacement(loader, K=400, seed=0)
