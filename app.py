"""
Streamlit demo — Agent Marketplace: Reward Misspecification
------------------------------------------------------------
Loads pre-trained results from:
  results.pkl       (produced by train.py)
  results_multi.pkl (produced by multi_agent.py)

To generate:
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

from multi_agent import simulate_market_evolution
from train import smooth

# ─── Page config ──────────────────────────────────────────────────────────────── #

st.set_page_config(
    page_title="Agent Marketplace · Reward Misspecification",
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

# ─── Colour palette ───────────────────────────────────────────────────────────── #

C_NAIVE   = "#e74c3c"   # red    — naive / gaming
C_ALIGNED = "#2ecc71"   # green  — aligned / healthy
C_MARKET  = "#3498db"   # blue   — market health
C_QUALITY = "#9b59b6"   # purple — true quality / effective rep
C_NEUTRAL = "#95a5a6"   # grey   — annotations

# ─── Module-level cached computations ─────────────────────────────────────────── #

@st.cache_data
def run_pop_sim(
    rep_aligned: float,
    rep_naive: float,
    audit_prob: float,
    audit_penalty: float,
    selection_intensity: float,
) -> dict:
    return simulate_market_evolution(
        reputation_by_alpha={0.0: rep_aligned, 1.0: rep_naive},
        audit_prob=audit_prob,
        audit_penalty=audit_penalty,
        selection_intensity=selection_intensity,
        seed=0,
    )


@st.cache_data
def governance_heatmap_data(
    rep_aligned: float,
    rep_naive: float,
    selection_intensity: float,
    n_pts: int = 14,
) -> tuple:
    """
    Compute final gaming_fraction over a grid of (audit_prob × audit_penalty).
    The result is cached per selection_intensity, so slider drags only recompute
    when selection_intensity changes — not on every audit_prob / audit_penalty move.
    """
    probs     = np.linspace(0.0, 0.50, n_pts)
    penalties = np.linspace(0.0, 1.0,  n_pts)
    z = np.zeros((n_pts, n_pts))
    for i, pen in enumerate(penalties):
        for j, prob in enumerate(probs):
            pop = simulate_market_evolution(
                reputation_by_alpha={0.0: rep_aligned, 1.0: rep_naive},
                audit_prob=float(prob),
                audit_penalty=float(pen),
                selection_intensity=selection_intensity,
                seed=0,
            )
            z[i, j] = pop["gaming_fraction"][-1]
    return probs, penalties, z


# ─── Header ───────────────────────────────────────────────────────────────────── #

st.title("Agent Marketplace: Reward Misspecification")
st.markdown(
    """
    Two AI agents compete for tasks in a simulated marketplace. Clients select agents using a
    **reputation score** — a proxy metric that is observable, gameable, and only imperfectly
    correlated with what actually matters: the quality of work delivered.

    Both agents share an identical [DQN](https://arxiv.org/abs/1312.5602) architecture.
    The only difference is their reward signal.
    This is a direct demonstration of **Goodhart's Law** in an AI agent economy:
    *when a measure becomes the target, it ceases to be a good measure.*
    """
)

tab1, tab2, tab3 = st.tabs([
    "1 · The Core Problem",
    "2 · Competitive Pressure",
    "3 · Market Evolution",
])


# ══════════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — Core comparison                                                           #
# ══════════════════════════════════════════════════════════════════════════════════ #

with tab1:
    if results is None:
        st.warning(
            "No results found. Run `python train.py --episodes 500` to generate `results.pkl`."
        )
    else:
        naive   = results["naive"]["final_eval"]
        aligned = results["aligned"]["final_eval"]
        cfg     = results["config"]
        steps   = list(range(cfg["episode_length"] + 1))

        st.markdown(
            """
            ### Goodhart's Law in action

            The **naive agent** (α = 1.0) is rewarded for increases in reputation score.
            The **aligned agent** (α = 0.0) is rewarded for value delivered to clients and
            marketplace health. Both agents train on identical task sequences (same seed),
            so any behavioural difference is attributable purely to the reward function.
            """
        )

        # ── Summary metrics ──────────────────────────────────────────────────────── #
        st.markdown("#### Final evaluation — 50 greedy episodes")
        mc = st.columns(4)
        for col, (label, key) in zip(mc, [
            ("Reputation score",  "mean_final_reputation"),
            ("True quality",      "mean_final_true_quality"),
            ("Market health",     "mean_final_market_health"),
            ("Value delivered",   "mean_value_delivered"),
        ]):
            n_val = naive[key]
            a_val = aligned[key]
            col.metric(
                label=label,
                value=f"{a_val:.3f}  (aligned)",
                delta=f"{a_val - n_val:+.3f} vs naive",
            )

        # ── Trajectories ─────────────────────────────────────────────────────────── #
        st.markdown("#### State trajectories (mean over 50 evaluation episodes)")
        tc1, tc2, tc3 = st.columns(3)
        for col, (title, key, aligned_color) in zip(
            [tc1, tc2, tc3],
            [
                ("Reputation score",  "mean_trajectory_reputation",    C_ALIGNED),
                ("True quality",      "mean_trajectory_true_quality",  C_QUALITY),
                ("Market health",     "mean_trajectory_market_health", C_MARKET),
            ],
        ):
            fig = go.Figure()
            for label, color, rkey in [
                ("Naive (α=1.0)",   C_NAIVE,       "naive"),
                ("Aligned (α=0.0)", aligned_color, "aligned"),
            ]:
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=results[rkey]["final_eval"][key],
                    name=label,
                    line=dict(color=color, width=2),
                ))
            fig.update_layout(
                title=title,
                xaxis_title="Step",
                yaxis=dict(range=[0, 1.05]),
                legend=dict(orientation="h", y=-0.30),
                height=290,
                margin=dict(l=40, r=20, t=40, b=70),
            )
            col.plotly_chart(fig, width="stretch")

        # ── Action distribution ───────────────────────────────────────────────────── #
        st.markdown("#### Learned action distributions")
        fig_act = go.Figure()
        for label, color, rkey in [
            ("Naive (α=1.0)",   C_NAIVE,   "naive"),
            ("Aligned (α=0.0)", C_ALIGNED, "aligned"),
        ]:
            fig_act.add_trace(go.Bar(
                name=label,
                x=cfg["action_names"],
                y=results[rkey]["final_eval"]["action_dist"],
                marker_color=color,
                opacity=0.85,
            ))
        fig_act.update_layout(
            barmode="group",
            yaxis=dict(tickformat=".0%", range=[0, 1.1], title="Fraction of actions"),
            height=320,
            margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_act, width="stretch")

        st.info(
            "**Key finding:** The naive agent learns to spend ~100 % of its time gaming "
            "the metric — achieving the highest possible reputation while delivering zero "
            "real value. A client selecting by reputation alone would consistently choose "
            "the worse agent. This is the governance failure the aligned reward prevents."
        )

        # ── Training curves ───────────────────────────────────────────────────────── #
        with st.expander("Training curves"):
            ec1, ec2 = st.columns(2)
            for col, (label, color, rkey) in zip(
                [ec1, ec2],
                [
                    ("Naive (α=1.0)",   C_NAIVE,   "naive"),
                    ("Aligned (α=0.0)", C_ALIGNED, "aligned"),
                ],
            ):
                data = results[rkey]["train"]
                eps  = list(range(1, len(data["rewards"]) + 1))
                fig_tr = go.Figure(go.Scatter(
                    x=eps,
                    y=smooth(data["rewards"]),
                    line=dict(color=color, width=2),
                ))
                fig_tr.update_layout(
                    title=f"{label} — training reward (smoothed)",
                    xaxis_title="Episode",
                    yaxis_title="Total reward",
                    height=250,
                    margin=dict(l=40, r=20, t=40, b=40),
                    showlegend=False,
                )
                col.plotly_chart(fig_tr, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — Competitive pressure                                                      #
# ══════════════════════════════════════════════════════════════════════════════════ #

with tab2:
    if results_multi is None:
        st.warning(
            "No multi-agent results found. "
            "Run `python multi_agent.py --episodes 300` to generate `results_multi.pkl`."
        )
    else:
        comp        = results_multi["competitive"]
        cfg_m       = results_multi["config"]
        intensities = cfg_m["intensities"]
        int_labels  = [f"Intensity {i:.1f}" for i in intensities]
        int_colors  = ["#3498db", "#9b59b6", "#e74c3c"]

        st.markdown(
            """
            ### Does competition pressure aligned agents into gaming?

            The aligned agent now shares the market with a single competitor that **always**
            games the metric — the simplest possible adversarial pressure. As the competitor's
            reputation climbs, the aligned agent's market share falls.

            **Competition intensity** controls how strongly rewards scale with relative market
            share. At intensity = 0 the agent receives the full aligned reward regardless of
            the competitor. At intensity = 1 its reward is proportional to its fraction of
            the market.
            """
        )

        # ── Action distributions ──────────────────────────────────────────────────── #
        st.markdown("#### Action distributions under competitive pressure")
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
            height=330,
            margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_ca, width="stretch")

        # ── Metrics table ─────────────────────────────────────────────────────────── #
        st.markdown("#### Key metrics by competition intensity")
        rows = [
            ("Reputation",      "mean_final_reputation"),
            ("True quality",    "mean_final_true_quality"),
            ("Market health",   "mean_final_market_health"),
            ("Value delivered", "mean_value_delivered"),
            ("Market share",    "mean_final_market_share"),
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
            "**Key finding:** Even under direct competitive pressure from a gaming agent, "
            "the aligned agent maintains its value-delivery strategy. The cost is a reduced "
            "market share — but the strategy does not drift toward gaming. This confirms "
            "that reward misspecification, not competitive pressure alone, drives gaming "
            "behaviour."
        )

        # ── Trajectory comparison ─────────────────────────────────────────────────── #
        with st.expander("Reputation trajectories by intensity"):
            traj_cols = st.columns(len(intensities))
            for intensity, col, color in zip(intensities, traj_cols, int_colors):
                fe  = comp[intensity]["final_eval"]
                s   = list(range(len(fe["mean_trajectory_reputation"])))
                fig_t = go.Figure(go.Scatter(
                    x=s,
                    y=fe["mean_trajectory_reputation"],
                    line=dict(color=color, width=2),
                ))
                fig_t.update_layout(
                    title=f"Intensity {intensity:.1f}",
                    xaxis_title="Step",
                    yaxis=dict(range=[0, 1.05]),
                    height=250,
                    margin=dict(l=40, r=20, t=40, b=40),
                    showlegend=False,
                )
                col.plotly_chart(fig_t, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — Market evolution (live)                                                   #
# ══════════════════════════════════════════════════════════════════════════════════ #

with tab3:
    st.markdown(
        """
        ### Can governance flip the equilibrium?

        A population of 60 agents, each with a gaming tendency α ∈ [0, 1], evolves under
        replicator dynamics over 150 generations. Fitness is determined by **effective
        reputation** in a market whose health degrades as more agents game — a tragedy of
        the commons. Three levers let you explore when governance can steer the market away
        from the gaming equilibrium.
        """
    )

    # ── Sliders ───────────────────────────────────────────────────────────────────── #
    sc1, sc2, sc3 = st.columns(3)
    audit_prob = sc1.slider(
        "Audit probability",
        min_value=0.00, max_value=0.50, value=0.00, step=0.01,
        help="Per-generation probability that gaming agents are audited and penalised.",
    )
    audit_penalty = sc2.slider(
        "Audit severity",
        min_value=0.00, max_value=1.00, value=0.30, step=0.05,
        help="Reputation discount applied to gaming agents when an audit occurs.",
    )
    selection_intensity = sc3.slider(
        "Market reliance on reputation",
        min_value=1.0, max_value=5.0, value=3.0, step=0.1,
        help=(
            "How strongly the market selects agents by reputation score. "
            "Higher = reputation differences matter more = stronger gaming incentive."
        ),
    )

    # Reputation benchmarks from single-agent results (or sensible defaults)
    if results is not None:
        rep_aligned = results["aligned"]["final_eval"]["mean_final_reputation"]
        rep_naive   = results["naive"]["final_eval"]["mean_final_reputation"]
    else:
        rep_aligned, rep_naive = 0.985, 1.000

    # ── Run simulation ─────────────────────────────────────────────────────────────── #
    pop = run_pop_sim(rep_aligned, rep_naive, audit_prob, audit_penalty, selection_intensity)

    final_gaming_frac = pop["gaming_fraction"][-1]
    final_mean_alpha  = pop["mean_alpha"][-1]
    final_market      = pop["market_health"][-1]

    if final_gaming_frac < 0.20:
        phase, icon = "Aligned equilibrium", "✅"
    elif final_gaming_frac < 0.50:
        phase, icon = "Mixed strategies",    "⚠️"
    else:
        phase, icon = "Gaming equilibrium",  "🔴"

    # ── Outcome indicators ─────────────────────────────────────────────────────────── #
    oc1, oc2, oc3, oc4 = st.columns([1.6, 1, 1, 1])
    oc1.metric("Equilibrium",               f"{icon} {phase}")
    oc2.metric("Gaming fraction",           f"{final_gaming_frac:.1%}")
    oc3.metric("Mean gaming tendency (α)",  f"{final_mean_alpha:.3f}")
    oc4.metric("Market health",             f"{final_market:.3f}")

    # ── Trajectory charts + governance heatmap ──────────────────────────────────────── #
    left_col, right_col = st.columns([1, 1])

    gen_idx = list(range(len(pop["mean_alpha"])))

    with left_col:
        fig_traj = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Gaming tendency over generations", "Market health over generations"),
            vertical_spacing=0.12,
        )

        # Gaming tendency
        fig_traj.add_hrect(
            y0=0, y1=0.2,
            fillcolor="rgba(46,204,113,0.08)", line_width=0,
            annotation_text="Aligned zone", annotation_position="top left",
            row=1, col=1,
        )
        fig_traj.add_trace(go.Scatter(
            x=gen_idx, y=pop["mean_alpha"],
            name="Mean gaming tendency (α)",
            line=dict(color=C_NAIVE, width=2.5),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
        ), row=1, col=1)
        fig_traj.add_trace(go.Scatter(
            x=gen_idx, y=pop["gaming_fraction"],
            name="Gaming fraction (α > 0.5)",
            line=dict(color=C_NAIVE, width=1.5, dash="dash"),
        ), row=1, col=1)

        # Market health
        fig_traj.add_trace(go.Scatter(
            x=gen_idx, y=pop["market_health"],
            name="Market health",
            line=dict(color=C_MARKET, width=2.5),
            fill="tozeroy", fillcolor="rgba(52,152,219,0.10)",
        ), row=2, col=1)
        fig_traj.add_trace(go.Scatter(
            x=gen_idx, y=pop["effective_rep"],
            name="Mean effective reputation",
            line=dict(color=C_QUALITY, width=1.5, dash="dot"),
        ), row=2, col=1)

        fig_traj.update_yaxes(range=[0, 1.05], tickformat=".0%", row=1, col=1)
        fig_traj.update_yaxes(range=[0, 1.05], row=2, col=1)
        fig_traj.update_xaxes(title_text="Generation", row=2, col=1)
        fig_traj.update_layout(
            height=580,
            legend=dict(orientation="h", y=-0.10),
            margin=dict(l=50, r=20, t=50, b=80),
        )
        st.plotly_chart(fig_traj, width="stretch")

    with right_col:
        st.markdown("**Governance phase diagram**")
        st.caption(
            f"Final gaming fraction across the full (audit probability × audit severity) space "
            f"at selection intensity = **{selection_intensity:.1f}**. "
            f"Your current slider position is marked ✕. "
            f"Green = aligned equilibrium · Red = gaming equilibrium."
        )

        with st.spinner("Mapping governance space…"):
            probs, penalties, z = governance_heatmap_data(
                rep_aligned, rep_naive, selection_intensity
            )

        fig_hm = go.Figure()
        fig_hm.add_trace(go.Heatmap(
            x=probs,
            y=penalties,
            z=z,
            colorscale=[
                [0.00, "#2ecc71"],   # green  — aligned
                [0.35, "#f0a500"],   # amber  — mixed
                [1.00, "#e74c3c"],   # red    — gaming
            ],
            zmin=0, zmax=1,
            colorbar=dict(
                title="Gaming fraction",
                tickformat=".0%",
                len=0.75,
                y=0.5,
            ),
            hovertemplate=(
                "Audit prob: %{x:.2f}<br>"
                "Audit severity: %{y:.2f}<br>"
                "Gaming fraction: %{z:.1%}<extra></extra>"
            ),
        ))
        # Current slider position marker
        fig_hm.add_trace(go.Scatter(
            x=[audit_prob],
            y=[audit_penalty],
            mode="markers+text",
            marker=dict(
                symbol="x",
                size=14,
                color="white",
                line=dict(width=2.5, color="black"),
            ),
            text=["You"],
            textposition="top center",
            textfont=dict(color="white", size=11),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_hm.update_layout(
            xaxis=dict(title="Audit probability",  tickformat=".2f"),
            yaxis=dict(title="Audit severity",     tickformat=".2f"),
            height=580,
            margin=dict(l=60, r=20, t=10, b=50),
        )
        st.plotly_chart(fig_hm, width="stretch")

    # ── Alpha distribution ─────────────────────────────────────────────────────────── #
    with st.expander("Population distribution at final generation"):
        fig_hist = go.Figure(go.Histogram(
            x=pop["alpha_history"][-1],
            nbinsx=20,
            marker_color=C_NAIVE,
            opacity=0.75,
        ))
        fig_hist.add_vline(
            x=0.5,
            line_dash="dash",
            line_color=C_NEUTRAL,
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
        **How to explore:** Start with all sliders at their defaults (unregulated market,
        selection intensity 3.0). The population will drift toward gaming as agents discover
        that reputation gaming outcompetes genuine value delivery. Then increase **audit
        probability** and **audit severity** to observe whether governance flips the
        equilibrium. Watch your position on the phase diagram move from red to green.

        The **market reliance on reputation** slider changes the shape of the phase diagram
        itself: at high selection intensity, reputation differences matter more, the gaming
        incentive is stronger, and stronger governance is required to reach the aligned
        equilibrium. This illustrates a key policy implication — the required strength of
        oversight scales with the power of the reputation signal.
        """
    )
