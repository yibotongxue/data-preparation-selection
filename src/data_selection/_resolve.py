from __future__ import annotations

import importlib

from omegaconf import DictConfig, OmegaConf


def resolve_target(cfg: DictConfig) -> object:
    """Resolve a DictConfig with _target_ into an instantiated object.

    Recursively resolves nested DictConfig values that also have _target_.
    """
    d = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(d, dict)
    return _instantiate(d)


def _instantiate(d: dict) -> object:
    target = d.pop("_target_")
    mod_path, cls_name = target.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    resolved: dict[str, object] = {}
    for k, v in d.items():
        if isinstance(v, dict) and "_target_" in v:
            resolved[k] = _instantiate(v)
        elif isinstance(v, list):
            resolved[k] = [
                (
                    _instantiate(item)
                    if isinstance(item, dict) and "_target_" in item
                    else item
                )
                for item in v
            ]
        else:
            resolved[k] = v
    return cls(**resolved)
