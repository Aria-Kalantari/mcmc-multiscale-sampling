"""Interactive dashboard for the single-subdomain conditioned sampler."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcmc_multiscale.app_core import (  # noqa: E402
    MethodRunConfig,
    RunSummary,
    default_m5_methods,
    run_lu_svd_comparison,
    run_method,
    run_methods,
)
from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.sampler import ConditionedSamplerState  # noqa: E402


def _method_label(
    theta_p_method: str,
    conditioning_mode: str,
    rhs_mode: str,
    rho: float | None,
) -> str:
    if conditioning_mode == "soft":
        return f"soft_{rhs_mode}_rho_{float(rho):.0e}"
    if rhs_mode == "zero":
        return f"hard_zero_{theta_p_method}"
    return f"hard_data_{theta_p_method}"


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4e}"


def _summary_row(summary: RunSummary) -> dict[str, object]:
    return {
        "method": summary.label,
        "rho": "n/a" if summary.rho is None else f"{summary.rho:.0e}",
        "max_candidate_theta_norm": summary.max_candidate_theta_norm,
        "max_accepted_theta_norm": summary.max_accepted_theta_norm,
        "final_accepted_theta_norm": summary.final_accepted_theta_norm,
        "acceptance_rate": summary.acceptance_rate,
        "mean_residual": summary.mean_residual,
        "max_residual": summary.max_residual,
        "mean_interface_jump_accepted": summary.mean_interface_jump_accepted,
        "final_relative_k_error_accepted": summary.final_relative_k_error_accepted,
        "mean_hidden_null_norm": summary.mean_hidden_null_norm,
        "max_hidden_null_norm": summary.max_hidden_null_norm,
    }


def _plot_heatmap(field: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.8), constrained_layout=True)
    image = ax.imshow(field, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    fig.colorbar(image, ax=ax, shrink=0.82)
    st.pyplot(fig)
    plt.close(fig)


def _plot_lines(
    series: Sequence[tuple[str, Sequence[float]]],
    title: str,
    ylabel: str,
    expected_norm: float | None = None,
) -> None:
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    x_values = range(1, len(series[0][1]) + 1)
    for label, values in series:
        ax.plot(list(x_values), values, label=label)
    if expected_norm is not None:
        ax.axhline(expected_norm, color="black", linestyle="--", label="expected norm")
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    st.pyplot(fig)
    plt.close(fig)


def _values(
    states: Sequence[ConditionedSamplerState],
    attr: str,
) -> list[float]:
    return [float(getattr(state, attr)) for state in states]


def _optional_values(
    states: Sequence[ConditionedSamplerState],
    attr: str,
) -> list[float]:
    values: list[float] = []
    for state in states:
        value = getattr(state, attr)
        values.append(np.nan if value is None else float(value))
    return values


def _show_primary_run(
    states: list[ConditionedSamplerState],
    summary: RunSummary,
) -> None:
    final = states[-1]

    st.subheader("Acceptance Summary")
    cols = st.columns(4)
    cols[0].metric("Acceptance rate", f"{summary.acceptance_rate:.3f}")
    cols[1].metric("Accepted", f"{summary.accepted_count} / {summary.n_iter}")
    cols[2].metric("Max candidate norm", f"{summary.max_candidate_theta_norm:.3f}")
    cols[3].metric("Max accepted norm", f"{summary.max_accepted_theta_norm:.3f}")

    st.subheader("Field Heatmaps")
    heat_cols = st.columns(3)
    with heat_cols[0]:
        _plot_heatmap(final.G_accepted, "Final accepted log field G")
    with heat_cols[1]:
        _plot_heatmap(final.G_candidate, "Final candidate log field G")
    with heat_cols[2]:
        _plot_heatmap(final.pressure_accepted, "Final accepted pressure")

    st.subheader("Trace Diagnostics")
    _plot_lines(
        [
            ("candidate", _values(states, "theta_norm_candidate")),
            ("accepted", _values(states, "theta_norm_accepted")),
        ],
        "Theta norm",
        "norm",
        expected_norm=summary.expected_norm,
    )
    _plot_lines(
        [("candidate residual", _values(states, "constraint_residual_candidate"))],
        "Constraint residual",
        "relative residual",
    )
    _plot_lines(
        [
            ("candidate", _values(states, "interface_jump_candidate")),
            ("accepted", _values(states, "interface_jump_accepted")),
        ],
        "Interface jump",
        "RMS jump",
    )
    _plot_lines(
        [
            ("candidate", _optional_values(states, "relative_k_error_candidate")),
            ("accepted", _optional_values(states, "relative_k_error_accepted")),
        ],
        "Relative permeability error",
        "relative error",
    )

    st.subheader("Linear Algebra Diagnostics")
    diag_cols = st.columns(4)
    diag_cols[0].metric(
        "Hidden null mean", _fmt_optional(summary.mean_hidden_null_norm)
    )
    diag_cols[1].metric("Hidden null max", _fmt_optional(summary.max_hidden_null_norm))
    diag_cols[2].metric(
        "cond(A) mean/max", f"{summary.mean_cond_A:.3e} / {summary.max_cond_A:.3e}"
    )
    diag_cols[3].metric(
        "cond(B) mean/max",
        (
            "n/a"
            if summary.mean_cond_B is None
            else f"{summary.mean_cond_B:.3e} / {summary.max_cond_B:.3e}"
        ),
    )
    if summary.rho is not None:
        st.caption(f"Soft conditioning rho: {summary.rho:.3e}")

    st.subheader("Final Summary Table")
    st.dataframe([_summary_row(summary)], hide_index=True, use_container_width=True)


def _show_lu_svd_comparison(
    results: dict[str, tuple[list[ConditionedSamplerState], RunSummary]],
) -> None:
    lu_states, lu_summary = results["hard_data_lu"]
    svd_states, svd_summary = results["hard_data_svd"]
    tiny = np.finfo(np.float64).tiny

    st.subheader("LU vs SVD Comparison")
    cols = st.columns(2)
    cols[0].metric(
        "LU/SVD candidate max ratio",
        f"{lu_summary.max_candidate_theta_norm / max(svd_summary.max_candidate_theta_norm, tiny):.3f}",
    )
    cols[1].metric(
        "LU/SVD accepted max ratio",
        f"{lu_summary.max_accepted_theta_norm / max(svd_summary.max_accepted_theta_norm, tiny):.3f}",
    )
    _plot_lines(
        [
            ("LU candidate", _values(lu_states, "theta_norm_candidate")),
            ("SVD candidate", _values(svd_states, "theta_norm_candidate")),
        ],
        "LU vs SVD candidate theta norm",
        "norm",
        expected_norm=lu_summary.expected_norm,
    )
    _plot_lines(
        [
            ("LU accepted", _values(lu_states, "theta_norm_accepted")),
            ("SVD accepted", _values(svd_states, "theta_norm_accepted")),
        ],
        "LU vs SVD accepted theta norm",
        "norm",
        expected_norm=lu_summary.expected_norm,
    )


def _show_m5_comparison(
    results: dict[str, tuple[list[ConditionedSamplerState], RunSummary]],
) -> None:
    st.subheader("M5 Stability Fixes Comparison")
    summaries = [summary for _, summary in results.values()]
    st.dataframe(
        [_summary_row(summary) for summary in summaries],
        hide_index=True,
        use_container_width=True,
    )


def _run_dashboard() -> None:
    st.set_page_config(
        page_title="MCMC Multiscale Conditioning Dashboard",
        layout="wide",
    )
    st.title("MCMC Multiscale Conditioning Dashboard")
    st.write(
        "This app visualizes the single-subdomain conditioned sampler and compares "
        "stability behavior across LU, SVD, stabilized LU, zero-RHS, and "
        "soft-conditioning modes."
    )

    base_cfg = Config()
    with st.sidebar:
        st.header("Controls")
        Mb = st.selectbox("Mb", options=list(base_cfg.Mb_list), index=4)
        n_iter = st.number_input(
            "n_iter", min_value=1, max_value=1000, value=100, step=10
        )
        beta = st.slider("beta", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
        seed = st.number_input(
            "seed", min_value=0, max_value=2**31 - 1, value=7, step=1
        )
        theta_p_method = st.selectbox(
            "theta_p_method", options=["lu", "svd", "lu_stabilized"]
        )
        conditioning_mode = st.selectbox("conditioning_mode", options=["hard", "soft"])
        rhs_mode = st.selectbox("rhs_mode", options=["data", "zero"])
        rho_widget = st.number_input(
            "rho",
            min_value=1.0e-6,
            max_value=1.0e8,
            value=1.0e3,
            format="%g",
            disabled=conditioning_mode == "hard",
        )
        proposal = st.selectbox("proposal", options=["pcn", "random_walk"])
        run_comparison = st.checkbox("Run LU vs SVD comparison")
        run_m5 = st.checkbox("Run M5 stability fixes comparison")
        if n_iter > 300:
            st.warning("Large runs may take noticeably longer.")

        button_cols = st.columns(2)
        run_clicked = button_cols[0].button("Run", type="primary")
        reset_clicked = button_cols[1].button("Reset")

    if reset_clicked:
        for key in ("primary_result", "comparison_result", "m5_result", "error"):
            st.session_state.pop(key, None)
        st.rerun()

    if run_clicked:
        rho = float(rho_widget) if conditioning_mode == "soft" else None
        method = MethodRunConfig(
            label=_method_label(theta_p_method, conditioning_mode, rhs_mode, rho),
            theta_p_method=theta_p_method,
            conditioning_mode=conditioning_mode,
            rhs_mode=rhs_mode,
            rho=rho,
        )
        cfg = Config(seed=int(seed), beta=float(beta))
        try:
            with st.spinner("Running conditioned sampler..."):
                st.session_state["primary_result"] = run_method(
                    cfg=cfg,
                    method=method,
                    n_iter=int(n_iter),
                    Mb=int(Mb),
                    beta=float(beta),
                    seed=int(seed),
                    proposal=proposal,
                )
                if run_comparison:
                    st.session_state["comparison_result"] = run_lu_svd_comparison(
                        cfg=cfg,
                        n_iter=int(n_iter),
                        Mb=int(Mb),
                        beta=float(beta),
                        seed=int(seed),
                        proposal=proposal,
                    )
                else:
                    st.session_state.pop("comparison_result", None)

                if run_m5:
                    st.session_state["m5_result"] = run_methods(
                        cfg=cfg,
                        methods=default_m5_methods([1.0e1, 1.0e3]),
                        n_iter=int(n_iter),
                        Mb=int(Mb),
                        beta=float(beta),
                        seed=int(seed),
                        proposal=proposal,
                    )
                else:
                    st.session_state.pop("m5_result", None)
                st.session_state.pop("error", None)
        except Exception as exc:  # pragma: no cover - exercised manually in Streamlit
            st.session_state["error"] = str(exc)
            st.exception(exc)

    if "error" in st.session_state:
        st.error(st.session_state["error"])

    if "primary_result" not in st.session_state:
        st.info("Choose controls and click Run.")
        return

    states, summary = st.session_state["primary_result"]
    _show_primary_run(states, summary)

    if "comparison_result" in st.session_state:
        _show_lu_svd_comparison(st.session_state["comparison_result"])

    if "m5_result" in st.session_state:
        _show_m5_comparison(st.session_state["m5_result"])


if __name__ == "__main__":
    _run_dashboard()
