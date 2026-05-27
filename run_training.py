from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_config(path: str | Path) -> ModuleType:
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("large_scale_mfa_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Train large scale MFA from a simple config.")
    parser.add_argument("--config", required=True, help="Path to a Python config file.")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    from large_scale_mfa_pipeline import run_from_config

    run_from_config(cfg)


if __name__ == "__main__":
    main()
