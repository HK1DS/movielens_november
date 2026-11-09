"""
Dataset adapter registry.

config.yaml
  dataset:
    adapter: movielens          # a REGISTRY key, or "package.module:ClassName"
    options: {...}              # passed to the adapter constructor as keyword args

(The package is not called `datasets` on purpose: that name would shadow the
HuggingFace `datasets` package that sentence-transformers imports.)
"""
import importlib
import os

import yaml

from .base import DatasetAdapter, ItemMeta, MetaRelation, ProfileFacet  # noqa: F401

REGISTRY = {
    "movielens": "data_adapters.movielens:MovieLensAdapter",
    "csv": "data_adapters.generic_csv:GenericCSVAdapter",
}


def get_adapter(ds_cfg):
    spec = REGISTRY.get(ds_cfg["adapter"], ds_cfg["adapter"])
    if ":" not in spec:
        raise ValueError(f"unknown dataset adapter {ds_cfg['adapter']!r}; "
                         f"use one of {sorted(REGISTRY)} or 'package.module:ClassName'")
    mod, cls = spec.split(":", 1)
    return getattr(importlib.import_module(mod), cls)(**(ds_cfg.get("options") or {}))


def load_adapter(config_path=None):
    """Adapter from the `dataset:` section of the (run-specific) config.yaml."""
    path = config_path or os.environ.get("IKGR_CONFIG", "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "dataset" not in cfg:
        raise KeyError(f"{path} has no `dataset:` section")
    return get_adapter(cfg["dataset"])
