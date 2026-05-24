from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class CustomOmegaConfig(Generic[T]):
    """Lazy configuration that defers instantiation until .create().

    Wraps a class and its init parameters in a dict-like object.
    Params can be inspected and modified before calling .create().

    Nested CustomOmegaConfig values (including those in lists) are
    recursively resolved at create time.
    """

    def __init__(self, _cls: type[T], **_params: object) -> None:
        # Use object.__setattr__ to avoid triggering __setattr__ override
        object.__setattr__(self, "_CustomOmegaConfig__cls", _cls)
        object.__setattr__(self, "_CustomOmegaConfig__params", dict(_params))

    def __getattr__(self, name: str) -> object:
        if name.startswith("_CustomOmegaConfig__"):
            return object.__getattribute__(self, name)
        params = object.__getattribute__(self, "_CustomOmegaConfig__params")
        if name in params:
            return params[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        params = object.__getattribute__(self, "_CustomOmegaConfig__params")
        params[name] = value

    def __getitem__(self, key: str) -> object:
        return object.__getattribute__(self, "_CustomOmegaConfig__params")[key]

    def __setitem__(self, key: str, value: object) -> None:
        object.__getattribute__(self, "_CustomOmegaConfig__params")[key] = value

    def __repr__(self) -> str:
        cls = object.__getattribute__(self, "_CustomOmegaConfig__cls")
        params = object.__getattribute__(self, "_CustomOmegaConfig__params")
        return f"CustomOmegaConfig({cls.__name__}, {params})"

    @property
    def params(self) -> dict:
        return dict(object.__getattribute__(self, "_CustomOmegaConfig__params"))

    def merge(self, **overrides: object) -> CustomOmegaConfig[T]:
        """Return a new config with overrides applied."""
        params = dict(object.__getattribute__(self, "_CustomOmegaConfig__params"))
        params.update(overrides)
        cls = object.__getattribute__(self, "_CustomOmegaConfig__cls")
        new = CustomOmegaConfig(cls, **params)
        return new

    def create(self) -> T:
        """Recursively resolve and instantiate."""
        cls = object.__getattribute__(self, "_CustomOmegaConfig__cls")
        params = object.__getattribute__(self, "_CustomOmegaConfig__params")
        resolved: dict[str, object] = {}
        for k, v in params.items():
            if isinstance(v, CustomOmegaConfig):
                resolved[k] = v.create()
            elif isinstance(v, list):
                resolved[k] = [
                    item.create() if isinstance(item, CustomOmegaConfig) else item
                    for item in v
                ]
            else:
                resolved[k] = v
        return cls(**resolved)
