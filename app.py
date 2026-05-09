"""
Portfolio Streamlit demo — AI Agent Marketplace: Infrastructure Design
----------------------------------------------------------------------
Loads pre-trained results from:
  results.pkl       (produced by train.py — 5-alpha infrastructure sweep)
  results_multi.pkl (produced by multi_agent.py — competitive + population)

To regenerate:
    python train.py --episodes 500
    python multi_agent.py --episodes 300

Then run:
    streamlit run app.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from multi_agent import governance_efficiency_data, simulate_market_evolution
from train import smooth


# ─── Page config ─────────────────────────────────────────────────────────────── #

st.set_page_config(
    page_title="AI Agent Marketplace — Infrastructure Design",
    page_icon="📊",
    layout="wide",
)


# ─── Data loading ─────────────────────────────────────────────────────────────── #

@st.cache_data
def load_pkl(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


results       = load_pkl("results.pkl")
results_multi = load_pkl("results_multi.pkl")


# ─── Colour palette ──────────────────────────────────────────────────────────── #

C_GAMING  = "#e74c3c"   # red    — score manipulation / gaming
C_ALIGNED = "#2ecc71"   # green  — deep engagement / value delivery
C_MARKET  = "#3498db"   # blue   — market health / capability maintenance
C_SHALLOW = "#f0a500"   # amber  — shallow templating
C_QUALITY = "#9b59b6"   # purple — true quality
C_NEUTRAL = "#95a5a6"   # grey   — annotations

FPR_COLORS = ["#2ecc71", "#f0a500", "#e74c3c"]   # perfect → imprecise


# ─── Cached simulations ──────────────────────────────────────────────────────── #

@st.cache_data
def run_pop_sim(
    rep_aligned: float,
    rep_naive: float,
    audit_prob: float,
    audit_penalty: float,
    false_positive_rate: float,
    selection_intensity: float,
) -> dict:
    return simulate_market_evolution(
        reputation_by_alpha={0.0: rep_aligned, 1.0: rep_naive},
        audit_prob=audit_prob,
        audit_penalty=audit_penalty,
        false_positive_rate=false_positive_rate,
        selection_intensity=selection_intensity,
        seed=0,
    )


@st.cache_data
def run_efficiency_frontier(
    rep_aligned: float,
    rep_naive: float,
    selection_intensity: float,
    audit_penalty: float,
) -> dict:
    return governance_efficiency_data(
        reputation_by_alpha={0.0: rep_aligned, 1.0: rep_naive},
        selection_intensity=selection_intensity,
        audit_penalty=audit_penalty,
        seed=0,
    )


@st.cache_data
def governance_heatmap_data(
    rep_aligned: float,
    rep_naive: float,
    selection_intensity: float,
    false_positive_rate: float,
    n_pts: int = 14,
) -> tuple:
    """Gaming fraction over a grid of (audit_prob × audit_severity)."""
    probs     = np.linspace(0.0, 0.50, n_pts)
    penalties = np.linspace(0.0, 1.0,  n_pts)
    z = np.zeros((n_pts, n_pts))
    for i, pen in enumerate(penalties):
        for j, prob in enumerate(probs):
            pop = simulate_market_evolution(
                reputation_by_alpha={0.0: rep_aligned, 1.0: rep_naive},
                audit_prob=float(prob),
                audit_penalty=float(pen),
                false_positive_rate=false_positive_rate,
                selection_intensity=selection_intensity,
                seed=0,
            )
            z[i, j] = pop["gaming_fraction"][-1]
    return probs, penalties, z


# ─── Header ──────────────────────────────────────────────────────────────────── #

st.title("AI Agent Marketplace: Infrastructure Design")
st.markdown(
    """
    Gaming and value-misalignment in AI agent marketplaces is not a **reward specification**
    problem — it is an **infrastructure design** problem. Agents are rational: they respond
    to whatever the scoring system rewards. When infrastructure scores only on gameable proxy
    metrics, rational agents game those metrics. The fix is at the infrastructure level.

    Each simulation uses an identical [DQN](https://arxiv.org/abs/1312.5602) agent architecture
    trained under different infrastructure configurations. **The only variable is what the
    reputation system rewards.** Connects to Pillar 2 of *The Epistemic Gate* (Hallam, 2026).
    """
)

tab1, tab2, tab3 = st.tabs([
    "1 · Infrastructure Design",
    "2 · Selection Pressure",
    "3 · Governance Efficiency",
])


# ══════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — Infrastructure Design                                                  #
# ══════════════════════════════════════════════════════════════════════════════ #

with tab1:
    if results is None or 0.0 not in results:
        st.warning(
            "Results not found. Run `python train.py --episodes 500` to generate `results.pkl`."
        )
    else:
        cfg           = results["config"]
        alphas        = cfg["alphas"]
        alpha_labels  = cfg["alpha_labels"]
        action_names  = cfg["action_names"]
        action_colors = cfg["action_colors"]

        # Ordered from least to most governance: 1.0 → 0.0
        ordered_alphas = sorted(alphas, reverse=True)
        level_labels   = [alpha_labels[a][0] for a in ordered_alphas]

        st.markdown(
            """
            ### Infrastructure signal fidelity shapes rational behaviour

            The **infrastructure signal fidelity** (α) controls what the scoring system
            actually measures. At α = 1.0 the platform scores unverified proxy metrics
            only — client ratings, completion rates — which are cheap to game. At α = 0.0
            the platform independently verifies genuine value delivery.

            All agents share the same DQN architecture and face identical task sequences.
            Behavioural differences are attributable entirely to infrastructure design.
            """
        )

        # ── Strategy sweep: stacked bar ───────────────────────────────────────────── #
        st.markdown("#### Strategy adopted under each governance design")
        fig_sweep = go.Figure()
        for j, (aname, acolor) in enumerate(zip(action_names, action_colors)):
            fig_sweep.add_trace(go.Bar(
                name=aname,
                x=level_labels,
                y=[results[a]["final_eval"]["action_dist"][j] for a in ordered_alphas],
                marker_color=acolor,
                opacity=0.88,
            ))
        fig_sweep.update_layout(
            barmode="stack",
            xaxis_title="Governance design  (left = no governance  →  right = full quality verification)",
            yaxis=dict(tickformat=".0%", range=[0, 1.02], title="Action distribution"),
            legend=dict(orientation="h", y=1.10),
            height=340,
            margin=dict(l=40, r=20, t=60, b=50),
        )
        st.plotly_chart(fig_sweep, width="stretch")

        # ── Outcome sweep ─────────────────────────────────────────────────────────── #
        st.markdown("#### Market outcomes by governance design")
        fig_out = make_subplots(
            rows=1, cols=3,
            subplot_titles=("True quality", "Market health", "Value delivered"),
        )
        for col_idx, (metric_key, color) in enumerate([
            ("mean_final_true_quality",   C_QUALITY),
            ("mean_final_market_health",  C_MARKET),
            ("mean_value_delivered",      C_ALIGNED),
        ], start=1):
            fig_out.add_trace(go.Scatter(
                x=level_labels,
                y=[results[a]["final_eval"][metric_key] for a in ordered_alphas],
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=9),
                showlegend=False,
            ), row=1, col=col_idx)
        fig_out.update_yaxes(range=[0, 1.05])
        fig_out.update_layout(
            height=280,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_out, width="stretch")

        st.info(
            "**Infrastructure finding:** Under no governance (α = 1.0), rational agents "
            "converge to 100 % score manipulation — the strategy the infrastructure rewards. "
            "True quality, market health, and value delivered all collapse to zero. Under "
            "full quality verification (α = 0.0), the same agents engage exclusively in "
            "deep work. The difference is entirely attributable to infrastructure design, "
            "not agent objectives or architecture."
        )

        # ── Drill-down: explore a specific governance level ────────────────────────── #
        st.divider()
        st.markdown("#### Drill down: trajectories for a specific governance level")
        selected_label = st.select_slider(
            "Governance level",
            options=level_labels,
            value=level_labels[0],   # default: No governance
        )
        selected_alpha = ordered_alphas[level_labels.index(selected_label)]
        short_label, description = alpha_labels[selected_alpha]

        st.info(f"**{short_label} (α = {selected_alpha:.2f})** — {description}")

        fe    = results[selected_alpha]["final_eval"]
        steps = list(range(cfg["episode_length"] + 1))

        traj_cols = st.columns(3)
        for col, (title, key, color) in zip(traj_cols, [
            ("Reputation score",  "mean_trajectory_reputation",    C_NEUTRAL),
            ("True quality",      "mean_trajectory_true_quality",  C_QUALITY),
            ("Market health",     "mean_trajectory_market_health", C_MARKET),
        ]):
            fig_t = go.Figure(go.Scatter(
                x=steps,
                y=fe[key],
                line=dict(color=color, width=2.5),
            ))
            fig_t.update_layout(
                title=title,
                xaxis_title="Step",
                yaxis=dict(range=[0, 1.05]),
                height=260,
                margin=dict(l=40, r=20, t=40, b=40),
                showlegend=False,
            )
            col.plotly_chart(fig_t, width="stretch")

        with st.expander("Training curve"):
            data = results[selected_alpha]["train"]
            eps  = list(range(1, len(data["rewards"]) + 1))
            clr  = C_GAMING if selected_alpha >= 0.75 else C_ALIGNED
            fig_tr = go.Figure(go.Scatter(
                x=eps,
                y=smooth(data["rewards"]),
                line=dict(color=clr, width=2),
            ))
            fig_tr.update_layout(
                xaxis_title="Episode",
                yaxis_title="Training reward (smoothed)",
                height=250,
                margin=dict(l=40, r=20, t=20, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_tr, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — Selection Pressure                                                     #
# ══════════════════════════════════════════════════════════════════════════════ #

with tab2:
    if results_multi is None:
        st.warning(
            "Multi-agent results not found. "
            "Run `python multi_agent.py --episodes 300` to generate `results_multi.pkl`."
        )
    else:
        comp        = results_multi["competitive"]
        cfg_m       = results_multi["config"]
        intensities = cfg_m["intensities"]
        int_labels  = [f"Intensity {i:.1f}" for i in intensities]
        int_colors  = [C_MARKET, C_QUALITY, C_GAMING]

        st.markdown(
            """
            ### Selection pressure is the mechanism, not the cause

            These agents all operate under **aligned infrastructure** (α = 0.0): the scoring
            system credits genuine value delivery. Each competes against a single opponent
            that always score-manipulates — the simplest possible competitive pressure. As
            the competitor's reputation climbs, the aligned agent's market share falls.

            **Competition intensity** controls how strongly market share loss cuts into the
            agent's reward. The question: does selection pressure push aligned agents toward
            gaming?

            This tab illustrates how infrastructure design propagates to population-level
            outcomes: selection pressure amplifies the strategies that infrastructure has
            already shaped. It does not create new ones.
            """
        )

        # ── Action distributions ───────────────────────────────────────────────────── #
        st.markdown("#### Action distribution under competitive pressure")
        fig_ca = go.Figure()
        for intensity, color, ilabel in zip(intensities, int_colors, int_labels):
            fig_ca.add_trace(go.Bar(
                name=ilabel,
                x=cfg_m["action_names"],
                y=comp[intensity]["final_eval"]["action_dist"],
                marker_color=color,
                opacity=0.85,
            ))
        fig_ca.update_layout(
            barmode="group",
            yaxis=dict(tickformat=".0%", range=[0, 1.1], title="Fraction of actions"),
            legend=dict(orientation="h", y=1.10),
            height=330,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_ca, width="stretch")

        # ── Metrics table ─────────────────────────────────────────────────────────── #
        st.markdown("#### Key outcomes by competition intensity")
        rows = [
            ("Reputation score",  "mean_final_reputation"),
            ("True quality",      "mean_final_true_quality"),
            ("Market health",     "mean_final_market_health"),
            ("Value delivered",   "mean_value_delivered"),
            ("Market share",      "mean_final_market_share"),
        ]
        df_comp = pd.DataFrame(
            {
                ilabel: [comp[i]["final_eval"][key] for _, key in rows]
                for i, ilabel in zip(intensities, int_labels)
            },
            index=[r for r, _ in rows],
        )
        st.dataframe(df_comp.style.format("{:.3f}"), width="stretch")

        st.info(
            "**Selection pressure finding:** Even under direct competition from a gaming "
            "agent at full market intensity, agents trained under aligned infrastructure "
            "maintain genuine value delivery. Market share falls — that is the cost of "
            "alignment under misaligned market conditions — but strategy does not drift "
            "toward gaming. Selection pressure amplifies existing strategies; "
            "infrastructure design determines what those strategies are."
        )

        # ── Trajectory comparison ─────────────────────────────────────────────────── #
        with st.expander("Reputation and quality trajectories by intensity"):
            traj_cols = st.columns(len(intensities))
            for intensity, col, color in zip(intensities, traj_cols, int_colors):
                fe  = comp[intensity]["final_eval"]
                s   = list(range(len(fe["mean_trajectory_reputation"])))
                fig_t = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=("Reputation", "True quality"),
                    vertical_spacing=0.15,
                )
                fig_t.add_trace(go.Scatter(
                    x=s, y=fe["mean_trajectory_reputation"],
                    line=dict(color=color, width=2), showlegend=False,
                ), row=1, col=1)
                fig_t.add_trace(go.Scatter(
                    x=s, y=fe["mean_trajectory_true_quality"],
                    line=dict(color=C_QUALITY, width=2), showlegend=False,
                ), row=2, col=1)
                fig_t.update_yaxes(range=[0, 1.05])
                fig_t.update_xaxes(title_text="Step", row=2, col=1)
                fig_t.update_layout(
                    title=f"Intensity {intensity:.1f}",
                    height=340,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                col.plotly_chart(fig_t, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — Governance Efficiency                                                  #
# ══════════════════════════════════════════════════════════════════════════════ #

with tab3:
    st.markdown(
        """
        ### Can governance flip the equilibrium? Precision matters more than intensity.

        A population of 60 agents evolves under replicator dynamics over 150 generations.
        Governance (periodic audits) penalises gaming agents — but audits carry a
        **false positive rate**: with some probability, an aligned agent is incorrectly
        penalised. This models a key governance trade-off: higher audit intensity catches
        more gaming, but imprecise audits impose collateral cost on legitimate agents.

        The **governance efficiency frontier** shows equilibrium market health as a function
        of audit intensity, for three levels of audit precision. The central finding:
        beyond a threshold, imprecise auditing fails to improve — or worsens — outcomes.
        Infrastructure operators should target *precision* over *intensity*.
        """
    )

    # ── Sliders ───────────────────────────────────────────────────────────────────── #
    sc1, sc2, sc3, sc4 = st.columns(4)
    audit_prob = sc1.slider(
        "Audit probability",
        min_value=0.00, max_value=0.50, value=0.00, step=0.01,
        help="Per-generation probability of an audit event.",
    )
    audit_penalty = sc2.slider(
        "Audit severity",
        min_value=0.00, max_value=1.00, value=0.30, step=0.05,
        help="Reputation discount applied to gaming agents when caught.",
    )
    false_positive_rate = sc3.slider(
        "Audit false positive rate",
        min_value=0.00, max_value=0.50, value=0.00, step=0.01,
        help=(
            "Probability that an aligned agent is incorrectly penalised. "
            "0 = perfect precision.  0.5 = random (no governance signal)."
        ),
    )
    selection_intensity = sc4.slider(
        "Market selection intensity",
        min_value=1.0, max_value=5.0, value=3.0, step=0.1,
        help="How strongly the market favours higher-reputation agents.",
    )

    # Reputation benchmarks
    if results_multi is not None and "reputation_by_alpha" in results_multi.get("config", {}):
        rep_aligned = results_multi["config"]["reputation_by_alpha"][0.0]
        rep_naive   = results_multi["config"]["reputation_by_alpha"][1.0]
    elif results is not None and 0.0 in results and 1.0 in results:
        rep_aligned = results[0.0]["final_eval"]["mean_final_reputation"]
        rep_naive   = results[1.0]["final_eval"]["mean_final_reputation"]
    else:
        rep_aligned, rep_naive = 0.985, 1.000

    # ── Run live simulation ────────────────────────────────────────────────────── #
    pop = run_pop_sim(
        rep_aligned, rep_naive,
        audit_prob, audit_penalty, false_positive_rate, selection_intensity,
    )

    final_gaming_frac = pop["gaming_fraction"][-1]
    final_mean_alpha  = pop["mean_alpha"][-1]
    final_market      = pop["market_health"][-1]

    if final_gaming_frac < 0.20:
        phase, icon = "Aligned equilibrium", "✅"
    elif final_gaming_frac < 0.50:
        phase, icon = "Mixed strategies",    "⚠️"
    else:
        phase, icon = "Gaming equilibrium",  "🔴"

    oc1, oc2, oc3, oc4 = st.columns([1.6, 1, 1, 1])
    oc1.metric("Equilibrium",              f"{icon} {phase}")
    oc2.metric("Gaming fraction",          f"{final_gaming_frac:.1%}")
    oc3.metric("Mean gaming tendency (α)", f"{final_mean_alpha:.3f}")
    oc4.metric("Market health",            f"{final_market:.3f}")

    # ── Main charts ───────────────────────────────────────────────────────────── #
    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.markdown("**Governance efficiency frontier**")
        st.caption(
            f"Equilibrium market health vs audit intensity at three audit precision levels "
            f"(severity = {audit_penalty:.2f}, selection intensity = {selection_intensity:.1f}). "
            f"Your current audit probability is marked ✕."
        )
        with st.spinner("Computing efficiency frontier…"):
            eff = run_efficiency_frontier(
                rep_aligned, rep_naive, selection_intensity, audit_penalty
            )

        fpr_vals   = sorted(eff["curves"].keys())
        fpr_labels = {
            0.00: "FPR = 0.00 (perfect precision)",
            0.15: "FPR = 0.15",
            0.30: "FPR = 0.30",
        }

        fig_eff = go.Figure()
        for fpr, color in zip(fpr_vals, FPR_COLORS):
            curve = eff["curves"][fpr]
            fig_eff.add_trace(go.Scatter(
                x=eff["audit_probs"],
                y=curve["market_health"],
                name=fpr_labels.get(fpr, f"FPR = {fpr:.2f}"),
                line=dict(color=color, width=2.5),
            ))
        fig_eff.add_trace(go.Scatter(
            x=[audit_prob],
            y=[final_market],
            mode="markers+text",
            marker=dict(symbol="x", size=14, color="white",
                        line=dict(width=2.5, color="black")),
            text=["You"],
            textposition="top center",
            textfont=dict(color="white", size=11),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_eff.update_layout(
            xaxis=dict(title="Audit probability", tickformat=".2f"),
            yaxis=dict(title="Market health at equilibrium", range=[0, 1.05]),
            legend=dict(orientation="h", y=1.10),
            height=420,
            margin=dict(l=50, r=20, t=60, b=50),
        )
        st.plotly_chart(fig_eff, width="stretch")

    with right_col:
        st.markdown("**Population dynamics — current settings**")
        gen_idx = list(range(len(pop["mean_alpha"])))

        fig_traj = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Gaming tendency (α)", "Market health"),
            vertical_spacing=0.12,
        )
        fig_traj.add_trace(go.Scatter(
            x=gen_idx, y=pop["mean_alpha"],
            name="Mean α",
            line=dict(color=C_GAMING, width=2.5),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.10)",
        ), row=1, col=1)
        fig_traj.add_trace(go.Scatter(
            x=gen_idx, y=pop["gaming_fraction"],
            name="Gaming fraction (α > 0.5)",
            line=dict(color=C_GAMING, width=1.5, dash="dash"),
        ), row=1, col=1)
        fig_traj.add_trace(go.Scatter(
            x=gen_idx, y=pop["market_health"],
            name="Market health",
            line=dict(color=C_MARKET, width=2.5),
            fill="tozeroy", fillcolor="rgba(52,152,219,0.10)",
        ), row=2, col=1)
        fig_traj.add_trace(go.Scatter(
            x=gen_idx, y=pop["effective_rep"],
            name="Effective reputation",
            line=dict(color=C_QUALITY, width=1.5, dash="dot"),
        ), row=2, col=1)

        fig_traj.update_yaxes(range=[0, 1.05], tickformat=".0%", row=1, col=1)
        fig_traj.update_yaxes(range=[0, 1.05], row=2, col=1)
        fig_traj.update_xaxes(title_text="Generation", row=2, col=1)
        fig_traj.update_layout(
            legend=dict(orientation="h", y=-0.12),
            height=420,
            margin=dict(l=50, r=20, t=50, b=80),
        )
        st.plotly_chart(fig_traj, width="stretch")

    # ── Phase diagram ─────────────────────────────────────────────────────────── #
    with st.expander("Governance phase diagram (audit probability × severity)"):
        st.caption(
            f"Final gaming fraction across the full (audit probability × severity) space "
            f"at selection intensity = {selection_intensity:.1f}, FPR = {false_positive_rate:.2f}. "
            f"Current position marked ✕."
        )
        with st.spinner("Mapping governance space…"):
            probs, penalties, z = governance_heatmap_data(
                rep_aligned, rep_naive, selection_intensity, false_positive_rate
            )

        fig_hm = go.Figure()
        fig_hm.add_trace(go.Heatmap(
            x=probs, y=penalties, z=z,
            colorscale=[
                [0.00, "#2ecc71"],
                [0.35, "#f0a500"],
                [1.00, "#e74c3c"],
            ],
            zmin=0, zmax=1,
            colorbar=dict(title="Gaming fraction", tickformat=".0%", len=0.75, y=0.5),
            hovertemplate=(
                "Audit prob: %{x:.2f}<br>"
                "Audit severity: %{y:.2f}<br>"
                "Gaming fraction: %{z:.1%}<extra></extra>"
            ),
        ))
        fig_hm.add_trace(go.Scatter(
            x=[audit_prob], y=[audit_penalty],
            mode="markers+text",
            marker=dict(symbol="x", size=14, color="white",
                        line=dict(width=2.5, color="black")),
            text=["You"], textposition="top center",
            textfont=dict(color="white", size=11),
            showlegend=False, hoverinfo="skip",
        ))
        fig_hm.update_layout(
            xaxis=dict(title="Audit probability", tickformat=".2f"),
            yaxis=dict(title="Audit severity",    tickformat=".2f"),
            height=480,
            margin=dict(l=60, r=20, t=10, b=50),
        )
        st.plotly_chart(fig_hm, width="stretch")

    # ── Population distribution ────────────────────────────────────────────────── #
    with st.expander("Population distribution at final generation"):
        fig_hist = go.Figure(go.Histogram(
            x=pop["alpha_history"][-1],
            nbinsx=20,
            marker_color=C_GAMING,
            opacity=0.75,
        ))
        fig_hist.add_vline(
            x=0.5, line_dash="dash", line_color=C_NEUTRAL,
            annotation_text="Gaming threshold (α = 0.5)",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            xaxis=dict(range=[0, 1], title="Gaming tendency (α)"),
            yaxis_title="Number of agents",
            height=270,
            margin=dict(l=50, r=20, t=20, b=50),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown(
        """
        ---
        **How to explore:** Start with all sliders at defaults (no audit, FPR = 0, selection
        intensity 3.0). The population drifts to gaming. Increase **audit probability** to
        observe governance counteract this. Then increase **false positive rate** to see how
        imprecise auditing erodes governance effectiveness — at sufficiently high FPR, even
        aggressive auditing fails to reach the aligned equilibrium. The efficiency frontier
        chart (left) shows this across the full range of audit intensities.

        **Key policy implication:** Infrastructure operators should prioritise audit
        *precision* over audit *intensity*. A lower-frequency, high-precision audit achieves
        better equilibrium outcomes than a high-frequency, imprecise one that penalises
        legitimate agents.
        """
    )
