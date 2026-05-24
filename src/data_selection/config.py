from __future__ import annotations

import importlib
from typing import Any

from omegaconf import DictConfig, OmegaConf


class CustomOmegaConfig[T](DictConfig):
    """OmegaConf DictConfig subclass with lazy _target_ instantiation.

    Python 3.12 native generic — .create() returns T.

    Usage:
        cfg = CustomOmegaConfig[RandomSelector](_target_="...", seed=42)
        selector: RandomSelector = cfg.create()
    """

    def create(self) -> T:
        """Recursively resolve _target_ and instantiate."""
        d = OmegaConf.to_container(self, resolve=True)
        assert isinstance(d, dict)
        return _instantiate(d)  # type: ignore[return-type]

    @classmethod
    def of(cls, _target_cls: type[T], **params: object) -> CustomOmegaConfig[T]:
        """Construct from a class reference plus params.

        Usage:
            cfg = CustomOmegaConfig.of(RandomSelector, seed=42)
        """
        d: dict[str, object] = {
            "_target_": f"{_target_cls.__module__}.{_target_cls.__qualname__}",
            **params,
        }
        return cls(d)


def _instantiate(d: dict[str, Any]) -> Any:
    target: str = d.pop("_target_")
    resolved: dict[str, Any] = {}
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

    mod_path, cls_name = target.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)(**resolved)


type MaybeConfig[T] = CustomOmegaConfig[T] | T | None


def maybe_create[T](cfg_or_obj: MaybeConfig[T]) -> T | None:
    """If the value is a CustomOmegaConfig, call .create(); otherwise return as-is."""
    if isinstance(cfg_or_obj, CustomOmegaConfig):
        return cfg_or_obj.create()
    return cfg_or_obj
