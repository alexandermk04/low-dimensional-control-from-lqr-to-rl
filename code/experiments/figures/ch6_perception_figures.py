"""Figures and tables for chapter 6's architecture comparison.

Run with::

    python -m experiments.figures.ch6_perception_figures
"""

import config as cfg
import settings
from experiments.figures.aggregate import (
    label_for,
    latest_per_experiment,
    tidy,
    with_suboptimality,
)
from experiments.figures.panels import (
    box_metric,
    complexity_bars,
    learning_bands,
    trajectory_overlay,
)
from experiments.figures.style import figure_dir, new_figure, save, save_variants, table_dir
from experiments.figures.tables import _fit, metric_table, write_table
from experiments.store import load_experiment

EXPERIMENT = "ch6_perception"

ORDER = tuple(settings.CH6_SOLVERS)

# Baseline scorer
REFERENCE = "oracle_coarse"

# Rows shown above and below the dot
PREVIEW_HALF_ROWS = 8


def observation_preview():
    """The two stacked frames the policy receives, cropped to the rows around the dot."""
    import numpy as np

    from experiments.registry import _pixel_setup
    from lqr_base.thruster_ring import ThrusterOraclePolicy
    from reinforcement_learning.pixel_env import build_pixel_env
    from seeding import sub_seed

    K, _, horizon, _ = _pixel_setup(None)
    env = build_pixel_env(sub_seed(0, "pixel-preview"), horizon=horizon)
    scene = env.config.scene

    # Largest shift between the two frames
    oracle = ThrusterOraclePolicy(K, env.c)
    observation, info = env.reset()
    shown, largest = observation, 0.0
    for _ in range(horizon):
        previous = info["state"][0]
        observation, _, _, _, info = env.step(oracle(info["state"]))
        if abs(info["state"][0] - previous) > largest:
            shown, largest = observation, abs(info["state"][0] - previous)

    middle = scene.height // 2
    rows = slice(middle - PREVIEW_HALF_ROWS, middle + PREVIEW_HALF_ROWS + 1)
    half_height = PREVIEW_HALF_ROWS * scene.metres_per_pixel

    figure, axes = new_figure(width=0.48, aspect=0.5, nrows=2, sharex=True)
    for ax, frame, label in zip(axes, shown, (r"$k-1$", r"$k$")):
        ax.imshow(frame[rows], cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                  extent=(-scene.span, scene.span, -half_height, half_height))
        ax.set_yticks([])
        ax.grid(False)
        ax.set_ylabel(label, rotation=0, labelpad=12, va="center")
    axes[0].tick_params(bottom=False)
    axes[-1].set_xticks([-scene.span, 0.0, scene.span])
    axes[-1].set_xlabel(r"position $p$ in $\mathrm{m}$")

    save(figure, "pixel_observation", EXPERIMENT)


def parameter_table(rows):
    """Trainable, frozen and total actor parameters per solver."""

    def count(n: int) -> str:
        return "$" + f"{n:,}".replace(",", r"\,") + "$"

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Solver & Trainable & Frozen & Total \\",
        r"\midrule",
    ]
    for solver in ORDER[1:]:
        # Same architecture for every seed
        row = next(r for r in rows if r["solver"] == solver)
        total = int(row["policy_complexity"])
        frozen = int(row.get("policy_complexity_frozen") or 0)
        lines.append(" & ".join(
            [label_for(solver), count(total - frozen), count(frozen), count(total)]
        ) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    write_table(
        "\n".join([
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Actor parameters of the PPO variants in the camera environment, "
            r"split into those trained by PPO and those frozen after pretraining.}",
            r"\label{tab:ch6-perception-parameters}",
            _fit("\n".join(lines)),
            r"\end{table}",
        ]),
        "ch6_perception_parameters",
        EXPERIMENT,
    )


def main():
    observation_preview()

    records = latest_per_experiment(load_experiment(EXPERIMENT))
    rows = with_suboptimality(tidy(records), reference=REFERENCE, by=("seed",))

    learned = [r for r in rows if r["solver"] != REFERENCE]
    learned_records = [r for r in records if r.solver != REFERENCE]

    ax = box_metric(
        learned, "solver", "training_subopt_final", order=ORDER, log="symlog"
    )
    save(ax.figure, "ch6_perception_training_subopt", EXPERIMENT)

    ax = learning_bands(learned_records, "solver", x="env_steps", order=ORDER)
    ax.set_xlabel("Agent steps")
    save(ax.figure, "ch6_perception_curves_steps", EXPERIMENT)

    # Use a seed which is available in every solver
    shared_seed = min(set.intersection(*(
        {r.seed for r in records if r.solver == solver} for solver in ORDER
    )))
    axes = trajectory_overlay(
        records, "solver", dt=cfg.DEFAULT_DT * cfg.DEFAULT_ACTION_REPEAT,
        seed=shared_seed, order=ORDER,
    )
    save(axes[0].figure, "ch6_perception_trajectories", EXPERIMENT)

    ax = box_metric(learned, "solver", "subopt", order=ORDER, log="symlog")
    save(ax.figure, "ch6_perception_subopt", EXPERIMENT)

    ax = complexity_bars(learned, "solver", order=ORDER)
    save(ax.figure, "ch6_perception_complexity", EXPERIMENT)

    for metric in ("r2_position", "r2_velocity", "r2_control"):
        ax = box_metric(learned, "solver", metric, order=ORDER, log=False)
        save(ax.figure, f"ch6_perception_{metric}", EXPERIMENT)

    # Half width figures
    for metric in ("terminated_rate", "null_waste_mean"):
        save_variants(
            lambda metric=metric, **size: box_metric(
                learned, "solver", metric, order=ORDER, log=False, **size
            ),
            f"ch6_perception_{metric}",
            EXPERIMENT,
        )

    parameter_table(learned)

    write_table(
        metric_table(
            rows, "solver",
            metrics=("J_final", "training_subopt_final", "subopt",
                     "r2_position", "r2_velocity", "r2_control", "latent_dim",
                     "off_screen_rate", "terminated_rate", "null_waste_mean",
                     "policy_complexity", "iterations", "total_env_steps", "best_env_steps", "total_time"),
            column_order=ORDER,
        ),
        "ch6_perception",
        EXPERIMENT,
    )

    write_table(
        metric_table(
            learned, "solver",
            metrics=("r2_position", "r2_velocity", "r2_control", "training_subopt_final"),
            column_order=ORDER,
            caption=r"Held-out $R^2$ of a linear probe from the latent state of each PPO variant to the true "
                    r"position, the true velocity and the control of the oracle, together with the final "
                    r"training suboptimality.",
            label="tab:ch6-perception-probe",
        ),
        "ch6_perception_probe",
        EXPERIMENT,
    )


if __name__ == "__main__":
    main()
