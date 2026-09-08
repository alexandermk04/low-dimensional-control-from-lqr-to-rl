"""
Helper functions to enforce that a specific setting has been set.
"""

import settings

__all__ = ["MissingSettingError", "require", "require_noise", "require_choice"]


class MissingSettingError(RuntimeError):
    """A setting the experiment depends on has not been decided yet."""


def require(value, name: str, set_by: str):
    """Return ``value``, or explain which experiment fixes it.

    Parameters
    ----------
    value : Any
        The value read from :mod:`settings`.
    name : str
        Its name in :mod:`settings`
    set_by : str
        Which experiment helps decide the value, e.g. ``"python -m experiments.ch4_sample_size"``.

    Returns
    -------
    Any
        ``value``, unchanged.

    Raises
    ------
    MissingSettingError
        If ``value`` is ``None``.
    """
    if value is None:
        raise MissingSettingError(
            f"settings.{name} is not set yet.\n"
            f"  Run: {set_by}"
        )
    return value


def require_noise(channels, levels=None) -> None:
    """Check that every noise intensity the run will need has been filled in.

    Parameters
    ----------
    channels : iterable of str
        Channels the experiment touches.
    levels : iterable of str or None, optional
        Levels it touches. Defaults to every entry.

    Raises
    ------
    MissingSettingError
        Lists every missing ``(channel, level)`` pair.
    KeyError
        If a channel or level is not known to :mod:`settings`.
    """
    levels = tuple(settings.NOISE_LEVELS if levels is None else levels)

    missing = []
    for channel in channels:
        table = settings.NOISE_STD[channel]
        for level in levels:
            if table[level] is None:
                missing.append(f"{channel}/{level}")

    if missing:
        raise MissingSettingError(
            "settings.NOISE_STD has undecided intensities: "
            + ", ".join(missing)
        )


def require_choice(value, name: str, allowed, set_by: str):
    """``require`` and check if the value is in an allowed set of choices."""
    value = require(value, name, set_by)

    if value not in allowed:
        raise MissingSettingError(
            f"settings.{name} = {value!r} is not one of {tuple(allowed)!r}."
        )

    return value
