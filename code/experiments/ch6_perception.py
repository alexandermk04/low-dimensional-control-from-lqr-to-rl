"""Chapter 6: perception-based control.

Run with::

    python -m experiments.ch6_perception
"""

import settings
from experiments.problem import RunSpec
from experiments.runner import run_grid

EXPERIMENT = "ch6_perception"


def build_specs() -> list[RunSpec]:
    """Architectures x seeds.
    """
    import config as cfg

    condition = {
        "observation": "pixels",
        "frame_size": cfg.DEFAULT_FRAME_SIZE,
        "frame_stack": cfg.DEFAULT_FRAME_STACK,
        "action_repeat": cfg.DEFAULT_ACTION_REPEAT,
        "n_thrusters": cfg.DEFAULT_N_THRUSTERS,
    }

    return [
        RunSpec(
            experiment=EXPERIMENT,
            solver=solver,
            seed=seed,
            inflation=None,
            iterations=settings.SOLVER_MAX_ITERATIONS,
            condition=condition,
        )
        for solver in settings.CH6_SOLVERS
        for seed in settings.SEEDS
    ]


def main():
    """Run the grid."""
    import config as cfg
    from reinforcement_learning.solvers import pixel_presets

    run_grid(
        build_specs(),
        EXPERIMENT,
        used_settings={
            "solvers": settings.CH6_SOLVERS,
            "seeds": settings.SEEDS,
            "frame_size": cfg.DEFAULT_FRAME_SIZE,
            "frame_stack": cfg.DEFAULT_FRAME_STACK,
            "action_repeat": cfg.DEFAULT_ACTION_REPEAT,
            "view_scale": cfg.DEFAULT_VIEW_SCALE,
            "n_thrusters": cfg.DEFAULT_N_THRUSTERS,
            "thrust_scale": cfg.DEFAULT_THRUST_SCALE,
            "exit_penalty_factor": cfg.EXIT_PENALTY_FACTOR,
            "rl_timesteps": settings.RL_MAX_TIMESTEPS[settings.CH6_PIXEL_INFLATION],
            "train_horizon": settings.TRAIN_HORIZON,
            "probe_horizon": settings.PROBE_HORIZON,
            "learning_rates": {s: p.learning_rate for s, p in pixel_presets().items()},
            "convergence": settings.CONVERGENCE,
            "rl_eval_freq": settings.RL_EVAL_FREQ,
            "probe_episodes": settings.PROBE_EPISODES,
            "pretrain_steps": settings.PRETRAIN_STEPS,
            "pretrain_epochs": settings.PRETRAIN_EPOCHS,
            "pve_position_dim": settings.PVE_POSITION_DIM,
            "encoder_latent_dim": settings.CH6_ENCODER_LATENT_DIM,
            "pve_loss_weights": settings.PVE_LOSS_WEIGHTS,
        },
    )


if __name__ == "__main__":
    main()
