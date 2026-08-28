"""Experiment configurations turned into JSONs.

:func:`to_jsonable` handles all of it, and records the class name of anything it
unpacks so that a stored config can be reloaded later.
"""

import dataclasses
from typing import Any

import numpy as np

__all__ = ["to_jsonable"]

#: Arrays with more elements than this are summarized for the json file
MAX_INLINE_ARRAY_SIZE = 64


def _summarize_array(array: np.ndarray) -> dict[str, Any]:
    """Describe a large array without inlining it."""
    return {
        "__array__": True,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "frobenius_norm": float(np.linalg.norm(array)),
        "note": "stored in full in the .npz file",
    }


def to_jsonable(obj: Any, max_inline_array_size: int = MAX_INLINE_ARRAY_SIZE) -> Any:
    """Recursively convert an object into JSON-serialisable primitives.

    Parameters
    ----------
    obj : Any
        Anything: a settings dataclass, a plain settings object, an array, a
        seed sequence, or a nested container of those.
    max_inline_array_size : int, optional
        Arrays with at most this many elements are written out elementwise;
        larger ones are replaced by a shape/dtype/norm summary.

    Returns
    -------
    Any
        A structure of dicts, lists, strings, numbers, booleans and ``None``.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        if obj.size <= max_inline_array_size:
            return obj.tolist()
        return _summarize_array(obj)

    if isinstance(obj, np.random.SeedSequence):
        return {
            "__class__": "SeedSequence",
            "entropy": to_jsonable(obj.entropy),
            "spawn_key": list(obj.spawn_key),
            "n_children_spawned": obj.n_children_spawned,
        }

    # recursively unpack
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_inline_array_size) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v, max_inline_array_size) for v in obj]

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Field-by-field
        return {
            "__class__": type(obj).__name__,
            **{
                f.name: to_jsonable(getattr(obj, f.name), max_inline_array_size)
                for f in dataclasses.fields(obj)
            },
        }

    if hasattr(obj, "__dict__"):
        return {
            "__class__": type(obj).__name__,
            **{
                k: to_jsonable(v, max_inline_array_size)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            },
        }

    return repr(obj)
