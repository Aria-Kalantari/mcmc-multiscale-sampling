from __future__ import annotations

from argparse import Namespace

from experiments.exp08c_recovery_decision import (
    DecisionSummary,
    TrajectoryPoint,
    _resolved_profile,
    _trajectory_shape,
    _verdict,
)


def _summary(
    label: str,
    *,
    relative_k_error: float,
    tail_mean_misfit: float,
    coverage: float,
    endpoint_max_r_hat: float,
    trajectory_shape: str = "plateau",
) -> DecisionSummary:
    return DecisionSummary(
        label=label,
        relative_k_error=relative_k_error,
        trajectory_shape=trajectory_shape,
        tail_mean_misfit=tail_mean_misfit,
        min_misfit=tail_mean_misfit,
        misfit_over_noise_floor=tail_mean_misfit / 32.0,
        coverage=coverage,
        acceptance_rate=0.25,
        endpoint_max_r_hat=endpoint_max_r_hat,
        best_max_r_hat=endpoint_max_r_hat,
        conservative_total_ess=20.0,
        total_forward_solves=1_000,
        wall_seconds=10.0,
        ess_per_1000_solves=20.0,
        ess_per_second=2.0,
        stopped_by="budget",
        updates_per_chain=(100, 100, 100, 100),
        scalar_summaries=(),
        trajectory=(),
    )


def test_decision_profile_is_opt_in_and_overridable() -> None:
    responsive = _resolved_profile(
        Namespace(
            decide=False,
            red_black_sweeps=None,
            global_iter=None,
            wall_cap_seconds=None,
            sample_stride=None,
            trajectory_points=None,
        )
    )
    decision = _resolved_profile(
        Namespace(
            decide=True,
            red_black_sweeps=None,
            global_iter=None,
            wall_cap_seconds=None,
            sample_stride=None,
            trajectory_points=None,
        )
    )
    overridden = _resolved_profile(
        Namespace(
            decide=True,
            red_black_sweeps=11,
            global_iter=12,
            wall_cap_seconds=13.0,
            sample_stride=2,
            trajectory_points=3,
        )
    )

    assert (responsive.red_black_sweeps, responsive.global_iter) == (2, 64)
    assert (decision.red_black_sweeps, decision.global_iter) == (500, 10_000)
    assert (
        overridden.red_black_sweeps,
        overridden.global_iter,
        overridden.wall_cap_seconds,
        overridden.sample_stride,
        overridden.trajectory_points,
    ) == (11, 12, 13.0, 2, 3)


def test_trajectory_shape_reports_plateau_or_descent() -> None:
    plateau = (
        TrajectoryPoint(10, 40, 0.70, 32.0, 1.0, 1.1),
        TrajectoryPoint(20, 80, 0.695, 32.0, 1.0, 1.1),
    )
    descent = (
        TrajectoryPoint(10, 40, 0.70, 32.0, 1.0, 1.1),
        TrajectoryPoint(20, 80, 0.65, 32.0, 1.0, 1.1),
    )

    assert _trajectory_shape(plateau) == "plateau"
    assert _trajectory_shape(descent) == "descending"


def test_trajectory_shape_uses_final_window_not_only_last_step() -> None:
    steady_descent = (
        TrajectoryPoint(10, 40, 0.60, 32.0, 1.0, 1.1),
        TrajectoryPoint(20, 80, 0.59, 32.0, 1.0, 1.1),
        TrajectoryPoint(30, 120, 0.582, 32.0, 1.0, 1.1),
        TrajectoryPoint(40, 160, 0.575, 32.0, 1.0, 1.1),
    )

    assert _trajectory_shape(steady_descent) == "descending"


def test_verdict_stays_ambiguous_until_both_schemes_converge() -> None:
    red_black = _summary(
        "posterior_red_black",
        relative_k_error=0.75,
        tail_mean_misfit=32.0,
        coverage=0.9,
        endpoint_max_r_hat=1.2,
    )
    global_pcn = _summary(
        "global_pcn",
        relative_k_error=0.6,
        tail_mean_misfit=32.0,
        coverage=0.9,
        endpoint_max_r_hat=1.0,
    )

    assert _verdict(red_black, global_pcn, noise_floor=32.0, level=0.9).startswith(
        "AMBIGUOUS"
    )


def test_verdict_reports_data_limited_for_similar_converged_floors() -> None:
    red_black = _summary(
        "posterior_red_black",
        relative_k_error=0.65,
        tail_mean_misfit=32.0,
        coverage=0.9,
        endpoint_max_r_hat=1.02,
    )
    global_pcn = _summary(
        "global_pcn",
        relative_k_error=0.6,
        tail_mean_misfit=32.0,
        coverage=0.88,
        endpoint_max_r_hat=1.03,
    )

    assert _verdict(red_black, global_pcn, noise_floor=32.0, level=0.9).startswith(
        "DATA-LIMITED"
    )


def test_verdict_reports_conditioning_defect_for_converged_plateau_gap() -> None:
    red_black = _summary(
        "posterior_red_black",
        relative_k_error=0.78,
        tail_mean_misfit=32.0,
        coverage=0.9,
        endpoint_max_r_hat=1.02,
    )
    global_pcn = _summary(
        "global_pcn",
        relative_k_error=0.6,
        tail_mean_misfit=32.0,
        coverage=0.9,
        endpoint_max_r_hat=1.03,
    )

    assert _verdict(red_black, global_pcn, noise_floor=32.0, level=0.9).startswith(
        "CONDITIONING DEFECT"
    )
