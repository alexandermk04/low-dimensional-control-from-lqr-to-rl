"""The four noise channels an LQR system in this repository can be exposed to.

Which point a channel attacks is the whole difference between them:

- **measurement** - additive, corrupts what the controller sees
- **control** - *multiplicative*, scaled by the commanded input, so a large
  control action is a noisier control action.
- **parametric** - perturbs the model matrices themselves, optionally set
  once per system (``time_invariant=True``) to model a mis-identified model
- **disturbance** - additive on the successor state, i.e. an external force.

Notes
-----
Noise is only injected while the runtime flag ``LQRBase.noise``.

All channels draw from the global :mod:`numpy.random` state, so seeding that
state makes a whole experiment reproducible.
"""

import numpy as np
from dataclasses import dataclass


class ParametricNoiseSettings:
    """Perturbation of the model matrices ``A`` and ``B``.

    Parameters
    ----------
    noise_std : float
        Standard deviation of the zero-mean Gaussian added to every entry of
        ``A`` and ``B``.
    time_invariant : bool
        If ``True``, one perturbation is drawn and then reused for the rest of
        the object's life

    Attributes
    ----------
    sampled_A, sampled_B : np.ndarray or None
        Cached perturbations, only populated when ``time_invariant`` is set.
    """

    noise_std: float
    time_invariant: bool

    sampled_A: np.ndarray | None
    sampled_B: np.ndarray | None

    def __init__(self, noise_std: float, time_invariant: bool):
        self.noise_std = noise_std
        self.time_invariant = time_invariant
        self.sampled_A = None
        self.sampled_B = None

    def reset(self):
        self.sampled_A = None
        self.sampled_B = None

    def sample_parametric_noise(self, A: np.ndarray, B: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw perturbations shaped like ``A`` and ``B``.

        Parameters
        ----------
        A : np.ndarray
            State matrix of shape ``(n, n)``; only its shape is used.
        B : np.ndarray
            Input matrix of shape ``(n, m)``; only its shape is used.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(noise_A, noise_B)``, to be *added* to the matrices by the caller.
            With ``time_invariant=True`` the same pair is returned on every call.
        """
        if self.time_invariant and self.sampled_A is not None and self.sampled_B is not None:
            return self.sampled_A, self.sampled_B

        noise_A = rng.normal(0, self.noise_std, size=A.shape)
        noise_B = rng.normal(0, self.noise_std, size=B.shape)

        if self.time_invariant:
            self.sampled_A = noise_A
            self.sampled_B = noise_B

        return noise_A, noise_B

class ControlNoiseSettings:
    """Multiplicative noise on the commanded control input.

    Parameters
    ----------
    noise_std : float
        Standard deviation of the relative error.
    """

    control_noise_std: float

    def __init__(self, noise_std: float):
        self.control_noise_std = noise_std

    def sample_control_noise(self, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw the additive term that realises a relative control error.

        Parameters
        ----------
        u : np.ndarray
            Commanded control input.

        Returns
        -------
        np.ndarray
            ``u * eps`` with ``eps ~ N(0, noise_std)`` drawn elementwise, so
            that ``u + result = u * (1 + eps)``.
        """

        control_noise = rng.normal(0, self.control_noise_std, size=u.shape)

        # Scaled noise based on the control input magnitude
        scaled_noise = u * control_noise

        return scaled_noise

class MeasurementNoiseSettings:
    """Additive noise on the observation handed to the policy.

    Parameters
    ----------
    noise_std : float
        Standard deviation of the zero-mean Gaussian added to each entry of
        the observation.
    """

    measurement_noise_std: float

    def __init__(self, noise_std: float):
        self.measurement_noise_std = noise_std

    def sample_measurement_noise(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw the additive measurement error.

        Parameters
        ----------
        x : np.ndarray
            The clean observation.

        Returns
        -------
        np.ndarray
            Noise of the same shape as the observation, to be added by the caller.
        """

        measurement_noise = rng.normal(0, self.measurement_noise_std, size=x.shape)

        return measurement_noise


class DisturbanceNoiseSettings:
    """External disturbance added to the successor state.

    Parameters
    ----------
    noise_std : float
        Magnitude of the disturbance - the standard deviation for the random
        types, and the amplitude for ``"sinusoidal"``.
    disturbance_type : {"gaussian", "sinusoidal", "constant"}
        - ``"gaussian"``  - newly drawn noise every step.
        - ``"sinusoidal"`` - a deterministic ``std * sin(0.1 * k)`` applied
          equally to every state entry, where ``k`` counts calls. Models a
          slow periodic drift.
        - ``"constant"``  - one random offset, drawn lazily on first use and
          held forever. Models a steady bias such as a wind.

    Attributes
    ----------
    const_val : np.ndarray or None
        Cached offset for the ``"constant"`` type.
    step_count : int
        Call counter driving the ``"sinusoidal"`` phase.
    """

    disturbance_noise_std: float
    disturbance_type: str

    const_val: np.ndarray | None
    step_count: int

    def __init__(self, noise_std: float, disturbance_type: str):
        self.disturbance_noise_std = noise_std
        self.disturbance_type = disturbance_type
        if disturbance_type not in {"gaussian", "sinusoidal", "constant"}:
            raise ValueError(f"Unknown disturbance type: {disturbance_type}")
        self.const_val = None
        self.step_count = 0

    def reset(self):
        """Reset the internal state of the noise generator.

        For ``"sinusoidal"``, resets the phase to zero. For ``"constant"``,
        forgets the cached offset so a new one is drawn on next use.
        """
        self.const_val = None
        self.step_count = 0

    def sample_disturbance(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw the disturbance to add to the successor state.

        Parameters
        ----------
        x : np.ndarray
            The successor state; only its shape is used.

        Returns
        -------
        np.ndarray
            Disturbance of the same shape.

        Raises
        ------
        ValueError
            If ``disturbance_type`` is not one of the three supported values.
        """
        if self.disturbance_type == "gaussian":
            disturbance = rng.normal(0, self.disturbance_noise_std, size=x.shape)
            return disturbance
        elif self.disturbance_type == "sinusoidal":
            disturbance = self.disturbance_noise_std * np.sin(0.01 * self.step_count) * np.ones_like(x)
            self.step_count += 1
            return disturbance
        elif self.disturbance_type == "constant":
            if self.const_val is None:
                self.const_val = rng.normal(0, self.disturbance_noise_std, size=x.shape)
            return self.const_val
        else:
            raise ValueError(f"Unknown disturbance type: {self.disturbance_type}")


@dataclass
class LQRNoiseSettings:
    """Bundle of the noise channels active on one LQR system.

    Every field defaults to ``None``, meaning that channel is switched off, so
    an experiment can isolate a single source of noise.

    Passing ``None`` in place of the whole bundle disables all four.

    Attributes
    ----------
    parametric_noise : ParametricNoiseSettings or None
    control_noise : ControlNoiseSettings or None
    measurement_noise : MeasurementNoiseSettings or None
    disturbance_noise : DisturbanceNoiseSettings or None
    """

    parametric_noise: ParametricNoiseSettings | None
    control_noise: ControlNoiseSettings | None
    measurement_noise: MeasurementNoiseSettings | None
    disturbance_noise: DisturbanceNoiseSettings | None

    def __init__(
        self,
        parametric_noise: ParametricNoiseSettings | None = None,
        control_noise: ControlNoiseSettings | None = None,
        measurement_noise: MeasurementNoiseSettings | None = None,
        disturbance_noise: DisturbanceNoiseSettings | None = None
    ):
        self.parametric_noise = parametric_noise
        self.control_noise = control_noise
        self.measurement_noise = measurement_noise
        self.disturbance_noise = disturbance_noise
