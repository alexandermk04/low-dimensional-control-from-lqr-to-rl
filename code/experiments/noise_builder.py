"""Turn the intensity combinations in :mod:`settings` into a noise settings bundle.
"""

import settings
from experiments.settings_guard import require_noise
from lqr_base.noise import (
    ControlNoiseSettings,
    DisturbanceNoiseSettings,
    LQRNoiseSettings,
    MeasurementNoiseSettings,
    ParametricNoiseSettings,
)

__all__ = ["build_noise", "noise_conditions"]


def build_noise(channels, level: str) -> LQRNoiseSettings | None:
    """Assemble the bundle for setting of the noise grid.

    Parameters
    ----------
    channels : str or iterable of str
        Which channels are active. A single name isolates one channel;
        ``settings.NOISE_CHANNELS`` activates all.
    level : str
        One of ``settings.NOISE_LEVELS``.

    Returns
    -------
    LQRNoiseSettings or None
        If ``None``, no noise can be applied.

    Raises
    ------
    experiments.settings_guard.MissingSettingError
        If any intensity this cell needs is still ``None``.
    """
    if isinstance(channels, str):
        channels = (channels,)
    channels = tuple(channels)

    if level == "none":
        return None

    require_noise(channels, levels=(level,))

    std = {channel: settings.NOISE_STD[channel][level] for channel in channels}

    return LQRNoiseSettings(
        parametric_noise=(
            ParametricNoiseSettings(
                std["parametric"], settings.PARAMETRIC_TIME_INVARIANT
            )
            if "parametric" in std else None
        ),
        control_noise=(
            ControlNoiseSettings(std["control"]) if "control" in std else None
        ),
        measurement_noise=(
            MeasurementNoiseSettings(std["measurement"])
            if "measurement" in std else None
        ),
        disturbance_noise=(
            DisturbanceNoiseSettings(std["disturbance"], settings.DISTURBANCE_TYPE)
            if "disturbance" in std else None
        ),
    )


def noise_conditions(levels=None, channels=None):
    """The full noise grid one experiment will be run on.

    Parameters
    ----------
    levels : iterable of str or None, optional
        Defaults to ``settings.NOISE_LEVELS``.
    channels : iterable of str or None, optional
        Defaults to ``settings.NOISE_CHANNELS``.

    Yields
    ------
    tuple[str, tuple[str, ...], str]
        ``(channel_name, active_channels, level)``, where ``channel_name`` is
        either one channel or ``"all"``. ``"none"`` is emitted once, under the
        name ``"clean"``, rather than once per channel.
    """
    levels = tuple(settings.NOISE_LEVELS if levels is None else levels)
    channels = tuple(settings.NOISE_CHANNELS if channels is None else channels)

    if "none" in levels:
        yield "clean", (), "none"

    for level in levels:
        if level == "none":
            continue
        for channel in channels:
            yield channel, (channel,), level
        yield "all", channels, level
