import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

def _cpu_state_dict(model):
    # Safe copy of weights to CPU (so we don’t pin VRAM)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

@torch.no_grad()
def _eval_nll_and_mse(model, loader, device):
    model.eval()
    tot_nll, tot_n = 0.0, 0
    for x, _ in loader:
        x = x.view(x.size(0), -1).to(device)
        nll = model.nll(x)                  # mean over batch
        B = x.size(0)
        tot_nll += nll.item() * B
        tot_n   += B
    return tot_nll / tot_n


import torch
from tqdm import tqdm

def train_nll(
    model,
    loader,
    *,
    val_loader=None,
    epochs=5,
    lr=1e-3,
    grad_clip=None,
    save_path=None,
    save_func=None,
    log_interval=100,           # number of batches between logging
    steps_per_epoch=None,       # optional cap for infinite/streaming loaders
):
    """
    Train with NLL, keep the best (lowest) NLL model.
    Works with streaming loaders that don't implement __len__.
    If `steps_per_epoch` is provided, we stop each epoch after that many steps.
    Otherwise we consume one pass over the iterable.
    """
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_metric = float("inf")
    best_state  = _cpu_state_dict(model)
    best_epoch  = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_nll, total_n = 0.0, 0

        # tqdm without total if unknown; with total if steps_per_epoch is set
        iterable = enumerate(loader, 1)
        pbar = tqdm(iterable, total=steps_per_epoch)

        for batch_idx, batch in pbar:
            # be agnostic to batch structure: accept x or (x, ...) or [x, ...]
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.view(x.size(0), -1).to(device)
            opt.zero_grad(set_to_none=True)
            nll = model.nll(x)     # mean over batch
            nll.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            B = x.size(0)
            total_nll += float(nll.item()) * B
            total_n   += B

            # log at intervals (by steps)
            if (batch_idx % log_interval) == 0:
                avg_so_far = total_nll / max(1, total_n)
                pbar.set_description(f"Epoch {ep:02d} | Step {batch_idx:06d} Train NLL={avg_so_far:.6f}")

            # for truly infinite streams, stop after steps_per_epoch if given
            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break

            # free ASAP
            del x, nll

        # guard against empty epoch (e.g., empty stream)
        if total_n == 0:
            avg_train_nll = float("nan")
        else:
            avg_train_nll = total_nll / total_n

        # validation (keep as-is; assumes _eval_nll_and_mse can iterate val_loader once)
        if val_loader is not None:
            val_nll = _eval_nll_and_mse(model, val_loader, device)
            select_metric = val_nll
        else:
            val_nll, val_mse = float("nan"), float("nan")
            select_metric = avg_train_nll

        improved = (select_metric < best_metric) if not (torch.isnan(torch.tensor(select_metric))) else False
        if improved:
            best_metric = select_metric
            best_state  = _cpu_state_dict(model)
            best_epoch  = ep
            if save_path and save_func:
                save_func(model, save_path)

        print(
            f"[epoch {ep:02d}] "
            f"train NLL={avg_train_nll:.6f}  "
            f"val NLL={val_nll:.6f} "
            f"{'** best **' if improved else ''}"
        )

    # restore best state before returning
    model.load_state_dict(best_state)
    print(f"Restored best model from epoch {best_epoch:02d} with metric={best_metric:.6f}")

    return dict(best_epoch=best_epoch, best_metric=best_metric)




@torch.no_grad()
def _eval_iso(model, loader, device, *, tau: float, lam_sparsity: float):
    model.eval()
    total_sum = recon_sum = sparse_sum = 0.0
    total_n = 0

    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.view(x.size(0), -1).to(device)
        B = x.size(0)

        _, Ez, _, _, _, _ = model._core(x)       # (B,K,q)
        W = model._W()                           # (K,D,q)
        xhat = model.mu[None, :, :] + torch.einsum("kdq,bkq->bkd", W, Ez)
        sq_errs = ((x[:, None, :] - xhat) ** 2).sum(dim=-1)  # (B,K)
        alpha = F.softmax(-sq_errs / float(tau), dim=1)

        loss_recon = (alpha * sq_errs).sum(dim=1).mean()
        l1_z = Ez.abs().sum(dim=-1)
        loss_sparse = float(lam_sparsity) * (alpha * l1_z).sum(dim=1).mean()
        loss_total = loss_recon + loss_sparse

        total_sum += float(loss_total.item()) * B
        recon_sum += float(loss_recon.item()) * B
        sparse_sum += float(loss_sparse.item()) * B
        total_n += B

    if total_n == 0:
        return float("nan"), float("nan"), float("nan")
    return total_sum / total_n, recon_sum / total_n, sparse_sum / total_n


def train_isotropic_reconstruction(
    model,
    loader,
    *,
    val_loader=None,
    epochs=10,
    lr=1e-3,
    tau=0.1,
    lam_sparsity=0.0,
    grad_clip=None,
    save_path=None,
    save_func=None,
    log_interval=100,
    steps_per_epoch=None,
    device=None,
):
    """
    Same training objective as your original isotropic reconstruction loop,
    plus:
      - optional val_loader
      - keep best (lowest) metric (val total if provided else train total)
      - optional save_func(model, save_path) when improved
      - restore best weights at end

    NO guessing about which params are trained:
      trains exactly: mu, dir_raw, scale_rho  (your MFA parameterization of W)
    """
    # choose device
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    model.to(device)

    # no guessing: require exact params for your MFA
    for name in ("mu", "dir_raw", "scale_rho"):
        if not hasattr(model, name):
            raise ValueError(f"Expected model to have nn.Parameter '{name}'")

    opt = torch.optim.Adam(
        [{"params": model.mu}, {"params": model.dir_raw}, {"params": model.scale_rho}],
        lr=lr,
    )

    best_metric = float("inf")
    best_state = _cpu_state_dict(model)
    best_epoch = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_sum = recon_sum = sparse_sum = 0.0
        total_n = 0

        iterable = enumerate(loader, 1)
        pbar = tqdm(iterable, total=steps_per_epoch)

        for step, batch in pbar:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.view(x.size(0), -1).to(device)
            B = x.size(0)

            opt.zero_grad(set_to_none=True)

            _, Ez, _, _, _, _ = model._core(x)       # (B,K,q)
            W = model._W()                           # (K,D,q)
            xhat = model.mu[None, :, :] + torch.einsum("kdq,bkq->bkd", W, Ez)

            sq_errs = ((x[:, None, :] - xhat) ** 2).sum(dim=-1)  # (B,K)
            alpha = F.softmax(-sq_errs / float(tau), dim=1)

            loss_recon = (alpha * sq_errs).sum(dim=1).mean()
            l1_z = Ez.abs().sum(dim=-1)
            loss_sparse = float(lam_sparsity) * (alpha * l1_z).sum(dim=1).mean()
            loss_total = loss_recon + loss_sparse

            loss_total.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            total_sum += float(loss_total.item()) * B
            recon_sum += float(loss_recon.item()) * B
            sparse_sum += float(loss_sparse.item()) * B
            total_n += B

            if (step % log_interval) == 0:
                avg_total = total_sum / max(1, total_n)
                avg_recon = recon_sum / max(1, total_n)
                avg_sparse = sparse_sum / max(1, total_n)
                pbar.set_description(
                    f"Epoch {ep:02d} | Step {step:06d} "
                    f"Train Total={avg_total:.6f} Recon={avg_recon:.6f} L1={avg_sparse:.6f}"
                )

            if steps_per_epoch is not None and step >= steps_per_epoch:
                break

        train_total = (total_sum / total_n) if total_n > 0 else float("nan")
        train_recon = (recon_sum / total_n) if total_n > 0 else float("nan")
        train_sparse = (sparse_sum / total_n) if total_n > 0 else float("nan")

        if val_loader is not None:
            val_total, val_recon, val_sparse = _eval_iso(
                model, val_loader, device, tau=tau, lam_sparsity=lam_sparsity
            )
            select_metric = val_total
        else:
            val_total, val_recon, val_sparse = float("nan"), float("nan"), float("nan")
            select_metric = train_total

        improved = math.isfinite(select_metric) and (select_metric < best_metric)
        if improved:
            best_metric = select_metric
            best_state = _cpu_state_dict(model)
            best_epoch = ep
            if save_path and save_func:
                save_func(model, save_path)

        print(
            f"[epoch {ep:02d}] "
            f"train Total={train_total:.6f} (Recon={train_recon:.6f}, L1={train_sparse:.6f})  "
            f"val Total={val_total:.6f} (Recon={val_recon:.6f}, L1={val_sparse:.6f})  "
            f"{'** best **' if improved else ''}"
        )

    model.load_state_dict(best_state)
    print(f"Restored best model from epoch {best_epoch:02d} with metric={best_metric:.6f}")
    return dict(best_epoch=best_epoch, best_metric=best_metric)