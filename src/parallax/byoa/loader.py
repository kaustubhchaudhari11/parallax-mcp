"""Load a developer-local factory. The dashboard never calls this module."""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from parallax.byoa.contract import ByoaSpecError

MAX_FACTORY_BYTES = 1_000_000


def load_factory(reference: str, root: Path) -> Callable[..., object]:
    """Import ``relative.py:attribute`` from inside ``root``."""

    path_text, attribute = reference.split(":")
    root_resolved = root.resolve()
    candidate = (root_resolved / path_text).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ByoaSpecError("agent factory escapes the project directory")
    if candidate.suffix != ".py" or not candidate.is_file():
        raise ByoaSpecError("agent factory file is missing")
    if candidate.stat().st_size > MAX_FACTORY_BYTES:
        raise ByoaSpecError("agent factory file exceeds the size limit")
    if _inside(candidate, root_resolved / "src" / "parallax"):
        raise ByoaSpecError("agent factory cannot be the native PARALLAX package")
    module_name = "parallax_byoa_factory_" + str(abs(hash(candidate)))
    spec = importlib.util.spec_from_file_location(module_name, candidate)
    if spec is None or spec.loader is None:
        raise ByoaSpecError("agent factory could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _attribute(module, attribute)


def factory_fingerprint(reference: str, root: Path) -> str:
    """SHA-256 of the factory file bytes."""

    from parallax.capture import sha256_text

    path_text, _attribute = reference.split(":")
    file_path = (root.resolve() / path_text).resolve()
    return sha256_text(file_path.read_text(encoding="utf-8"))


def _attribute(module: ModuleType, name: str) -> Callable[..., object]:
    if not hasattr(module, name):
        raise ByoaSpecError("agent factory attribute is missing")
    value = getattr(module, name)
    if not callable(value):
        raise ByoaSpecError("agent factory attribute is not callable")
    return value


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent.resolve())
    except ValueError:
        return False
    return True
