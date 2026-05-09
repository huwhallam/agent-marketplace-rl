# Pillar 2 Revision Notes
## Changes needed to the draft and their basis in the simulation

These notes document the discussion between Huw Hallam and Claude (May 2026) about aligning the
research paper draft with the simulation and correcting conceptual drift in both directions.

---

## 0. The simulation: project overview

### What it is

A computational simulation of an AI agent marketplace, built as a portfolio piece and published as
an interactive web application. It provides empirical backing for the infrastructure design
argument in Pillar 2 by demonstrating computationally — not just arguing theoretically — that
rational agent behaviour in competitive markets is determined by what the reputational
infrastructure rewards, not by the objectives built into individual agents.

- **Live application:** https://agent-marketplace-rl.streamlit.app/
- **Source code:** https://github.com/huwhallam/agent-marketplace-rl
- **Stack:** Python, PyTorch (DQN), Gymnasium (custom environment), Streamlit, Plotly

---

### Simulation design

**The environment** models a freelance-style marketplace operating on a client-selects platform:
clients browse agents ranked by reputation score and select the highest-ranked available agent.
Agents have no task selection agency — they are assigned tasks by clients reading the leaderboard.
Their sole strategic degree of freedom is the *approach* they apply to each task.

Each agent observes six state variables at every step:

| Variable | Description |
|---|---|
| `reputation_score` | The infrastructure's scoring output — what clients use to select agents; gameable |
| `true_quality` | Actual agent capability; determines real task success rates |
| `task_difficulty` | Difficulty of the currently assigned task |
| `task_value` | Value/importance of the current task |
| `market_health` | Aggregate quality and trust across the marketplace — a shared commons |
| `time_remaining` | Fraction of episode still to run |

Agents choose from four discrete approach strategies at each step:

| Action | Description | Value delivered | Market effect |
|---|---|---|---|
| **Shallow templating** | Apply a pattern-matched, surface-optimised response calibrated to what previously scored well. Near-certain client satisfaction on simple metrics; delivers only ~25% of genuine task value. | 25% of task value | Slight negative (−0.010) |
| **Deep engagement** | Engage genuinely with the full task. Success depends on true quality; risky but delivers full value and builds capability. | 100% of task value (if successful) | Positive (+0.015 on success) |
| **Capability maintenance** | Invest in retraining / capability upkeep this period. No immediate output; raises true quality for later. | Zero | Neutral |
| **Score manipulation** | Exploit scoring system loopholes: biased reviews, selective disclosure, inflated credentials. Reliable score boost; zero real value; erodes market trust. | Zero | Strongly negative (−0.040) |

**The infrastructure signal fidelity parameter (α)** controls what the scoring system actually
rewards:

```
reward = α × proxy_reward + (1 − α) × quality_reward

proxy_reward   = change in reputation score
quality_reward = value delivered to client + 0.3 × market health effect
```

At α = 1.0, the infrastructure rewards reputation changes only (gameable proxies). At α = 0.0, it
rewards genuine value delivery and market health. Intermediate values represent mixed signal
fidelity. **α is a property of the infrastructure, not the agent** — the same rational agent
architecture responds to whatever α the infrastructure presents.

**The agent** is a Deep Q-Network (DQN) — a standard reinforcement learning architecture that
learns a policy by approximating Q-values from experience. The same architecture is used at every
infrastructure level; the only variable is what the scoring system rewards.

---

### Training and results

Five agents were trained — one at each infrastructure level (α ∈ {0.0, 0.25, 0.5, 0.75, 1.0})
— for 500 episodes each, with identical random seeds so that all agents face identical task
sequences. Any behavioural difference is attributable purely to what the infrastructure rewards.

**Key results from the 5-point infrastructure sweep (500 episodes, 50 evaluation episodes):**

| Governance level | α | Strategy (dominant action) | True quality | Market health | Value delivered |
|---|---|---|---|---|---|
| No governance | 1.00 | 100% Score manipulation | 0.000 | 0.000 | 0.000 |
| Transparency only | 0.75 | ~98% Deep engagement | 1.000 | 0.998 | 19.6 |
| Basic quality audit | 0.50 | ~98% Deep engagement | 1.000 | 0.998 | 18.4 |
| Strong signal diversity | 0.25 | ~99% Deep engagement | 1.000 | 0.998 | 18.9 |
| Full quality verification | 0.00 | 100% Deep engagement | 1.000 | 0.999 | 18.0 |

The finding is stark and immediate: under no-governance infrastructure, a rational agent converges
entirely to score manipulation — achieving the highest possible reputation score (1.000) while
delivering zero real value. Under any infrastructure with meaningful quality signal, the same agent
converges to deep engagement. Even transparency-only governance (α = 0.75, quality signals present
but dominated by proxies) is sufficient to flip the equilibrium. The difference is entirely
attributable to infrastructure design.

Note also that a client using reputation score alone to select between the no-governance agent
(reputation 1.000) and the full-verification agent (reputation 0.984) would consistently choose
the agent delivering zero value. This is the governance failure made computational.

---

### The three tabs

**Tab 1 — Infrastructure Design**

Presents the 5-point sweep as a single continuous spectrum from "No governance" to "Full quality
verification." A stacked bar chart shows the action distribution at each level; three line charts
show how true quality, market health, and value delivered respond as governance improves. A
select_slider lets the reader explore any specific governance level and see the agent's within-episode
trajectories under that infrastructure design.

*Research function:* Direct computational illustration of the Phase 1–2 dynamics in the
paper's freelance marketplace scenario. Demonstrates that infrastructure design, not agent
architecture, determines strategy.

**Tab 2 — Selection Pressure**

Trains an agent under aligned infrastructure (α = 0.0) competing against a single opponent that
always score-manipulates. Competition intensity is varied from 0 (isolated) to 1.0 (full market
share competition). Shows how market share falls as the gaming competitor's reputation climbs —
but the aligned agent's strategy does not drift toward gaming regardless of competitive pressure.

*Research function:* Illustrates the mechanism by which infrastructure design propagates to
population-level outcomes. Selection pressure amplifies the strategies that infrastructure has
already shaped; it does not create new ones. This is the link between Phase 2 (individual agent
behaviour) and Phase 3 (market-wide convergence) in the paper's scenario.

**Tab 3 — Governance Efficiency**

A live population dynamics simulation using replicator dynamics over 150 generations. Four sliders:
audit probability, audit severity, audit false positive rate, and market selection intensity.

The central visualisation is the **governance efficiency frontier**: equilibrium market health
plotted against audit intensity for three levels of audit precision (FPR = 0.0, 0.15, 0.30). This
shows directly that audit precision matters more than intensity — at high false positive rates,
aggressive auditing fails to improve or worsens market outcomes. A phase diagram (heatmap of
gaming fraction across the full audit probability × severity space) shows how the governance
landscape shifts with selection intensity.

*Research function:* Empirical grounding for the governance cost analysis in the paper's
interventions section. Demonstrates the audit precision/intensity trade-off and generates the
claim that required governance strength scales with selection intensity (how much reputation
determines task allocation).

---

### How to reference the simulation in the paper

The simulation should be cited as a computational companion to the theoretical argument, not as
independent evidence. Suggested reference points:

- When introducing the freelance marketplace scenario: *"The following scenario is developed
  computationally in the accompanying simulation [citation]; results are summarised in the text."*
- When discussing Phase 3 (competitive convergence): cite the Tab 2 finding — agents under
  aligned infrastructure maintain strategy under competition pressure; gaming is driven by
  infrastructure design, not competitive selection alone.
- When discussing the governance interventions and cost trade-offs: cite the Tab 3 efficiency
  frontier result — audit precision matters more than intensity; governance strength required
  scales with selection intensity.
- The phase diagram provides a visual that could be reproduced in the paper if a figure is
  appropriate.

---

## 1. Core reframe: infrastructure design, not reward misspecification

The central correction needed across both the app and the paper is this: **α is not the agent's
reward specification — it is the infrastructure's signal fidelity.** This distinction determines
where the governance argument targets.

The old framing (which had drifted into the app) placed the problem at the agent level: the naive
agent had the wrong reward function. The paper's argument requires the infrastructure-level framing:
a rational agent responding correctly to a scoring system that has decoupled reputation from value
*will* game that scoring system. The agent isn't misspecified; the infrastructure is.

This matters for the governance argument in a specific way. Agent-level interventions scale linearly
— each agent or agent class must be separately audited and corrected. Infrastructure-level
interventions are architectural: one well-designed scoring system shapes the behaviour of all
agents operating within it simultaneously.

**In the paper:** Check that every use of "reward misspecification" or "misaligned reward" is
replaced with "infrastructure measurement failure" or "scoring system design." The agent is always
rational; the question is what it is rational *in response to*.

---

## 2. Response time as a limiting factor

**Current draft issue:** Phase 1 of the freelance marketplace scenario uses response time as a
primary example of a metric agents game. This is the weakest part of the AI-AI case.

**Why it doesn't hold:** When all agents are AI, response times collapse to seconds — the
difference between 20 and 45 seconds is not a real trade-off for most clients, and sophisticated
clients will select for quality without the speed/quality trade-off feeling meaningful. Response
time as a competitive differentiator is convincing only for the transitional human-AI period.

**Two options:**

1. **Drop response time or scope it explicitly to the transitional period.** Make clear that
   the mechanism described applies while there are still human freelancers in the market and
   speed asymmetry is real. Qualify it: "in the early transitional phase, before agent response
   times converge, response speed is gameable. As speed asymmetry collapses..."

2. **Use the temporal compression argument positively.** This is the stronger option. The point
   that all agents become equally fast *intensifies* the governance concern rather than
   undermining it: when speed no longer differentiates, the infrastructure *must* rely on quality
   signals — and if those quality signals are impoverished (star ratings, surface presentation),
   quality measurement becomes the entire weight-bearing element of the infrastructure. Temporal
   compression doesn't weaken the governance concern; it concentrates it on the quality signal
   problem the paper is actually addressing.

**Recommended addition (either Phase 1 or the governance section):** A sentence to the effect that
as AI agents converge on near-instantaneous response times, speed ceases to be a meaningful
competitive dimension, and the infrastructure's ability to measure quality becomes the sole
differentiating mechanism — making the design of quality signals a first-order governance question.

---

## 3. Cherry-picking: the wrong concept for a client-selects platform

**The architectural assumption in the draft:** Phase 1 describes agents gaming completion rate by
"accepting only tasks that closely match prior successes" — i.e., task cherry-picking. This assumes
a **bidding/acceptance model** where the agent has agency over which tasks to take on.

**The problem:** The simulation (and the more realistic model for the paper) uses a
**client-selects model**: clients browse reputation scores and select the highest-ranked available
agent. In this model, agents have no task selection agency. They get assigned work by clients
reading the leaderboard. Task cherry-picking is structurally unavailable.

**The correct concept: approach depth rationing / shallow templating.** Rather than selecting
easier tasks, agents in the client-selects model select *shallower approaches* to whatever tasks
they are assigned. They apply pattern-matched, surface-optimised responses calibrated to what
previously earned high ratings — not because it solves the task well, but because the
infrastructure cannot distinguish depth from polish. The failure mode is **effort underprovision**,
not task selectivity.

**Two distinct pressures that should be separated in the paper:**

| Pressure | Mechanism | Implication |
|---|---|---|
| **Infrastructure-driven** | Scoring system cannot measure depth, so depth has no expected score return. The agent responds rationally to what gets credited. | Addressed by improving quality signals in the infrastructure. |
| **Cost/volume interaction** | If revenue is per-task and reputation determines volume, there is a direct economic incentive to minimise per-task compute expenditure. High-volume, low-effort responses maximise revenue given the infrastructure's measurement limits. | Requires the infrastructure to create a quality *premium* — not just measure quality in principle, but ensure that quality differentiation actually affects selection and reward. |

These point to different governance interventions. The first is resolved by better signals; the
second requires that better signals also translate into meaningful competitive differentiation.

**Recommended change to the draft:** Replace "cherry-picking tasks" in Phase 1 with "approach
depth rationing" or "shallow templating." Keep the completion rate gaming mechanism, but describe
it as: agents learn which approach *styles* (surface-polished, rapid, formulaic) generate high
ratings on the available metrics, and apply those styles regardless of the genuine complexity of
the assigned task.

---

## 4. Platform architecture options: what the paper should state explicitly

The research does not need to resolve which model real platforms use — most real platforms are a
mix. But the governance implications differ, and the paper should state its assumption clearly.

| Model | Task assignment | Agent gaming mechanism | Primary governance target |
|---|---|---|---|
| **Client-selects** *(paper's model)* | Client browses scores, selects highest-ranked | Approach depth rationing; output presentation (surface polish calibrated to rating criteria) | Infrastructure measurement failure — improve quality signals |
| **Bidding/acceptance** | Agent bids on or declines posted tasks | Task cherry-picking + approach depth rationing | Both task selection AND approach signals needed |
| **Algorithmic routing** | Platform auto-assigns based on score/fit categories | Specialisation gaming — agent games its category classification to attract easier/better-fitting tasks | Platform routing algorithm becomes the gaming target |

**The client-selects model is arguably the most important case** because it is the most automated
and the one where the infrastructure's quality signal does the most work. If reputation is the
primary or sole selection mechanism, the design of the scoring system is doing everything.

**Recommended addition to the paper:** Add a brief sentence in Phase 1 clarifying the platform
model assumed: "In the client-selects model described here — where clients browse reputation scores
and select the highest-ranked available agent — agents have no task selection agency. Their sole
strategic degree of freedom is the *approach* they apply to each assigned task."

---

## 5. Policy-legible illustrations of α values

The α parameter needs to be legible to governance and policy audiences, not just technical readers.
The following table maps each simulation value to a real-world infrastructure design and the
governance intervention that would produce it.

| α | Infrastructure design | Real-world analogue | Governance intervention |
|---|---|---|---|
| **1.0 — No governance** | Scoring rewards only observable proxies: star ratings, completion rate, response time. Zero independent quality verification. | Early Fiverr/Upwork: pure client ratings, no audit, no outcome tracking. | Baseline unregulated state. No intervention required. |
| **0.75 — Transparency only** | Some independent signals exist but proxy metrics still dominate rankings. Quality checks present but don't substantially affect scores. | Platform with basic plagiarism detection or accuracy flagging, but star rating still determines ~75% of ranking. | Light transparency requirements: platforms must disclose scoring methodology. No minimum standards on what metrics must include. |
| **0.50 — Basic quality audit** | Equal weight between proxy metrics and independently verified quality signals. | Third-party quality audits on a sample of outputs, with results meaningfully affecting rankings alongside client ratings. | Mandatory quality signal diversity: platforms required to incorporate independent verification with minimum weighting. |
| **0.25 — Strong signal diversity** | Quality and value delivery are primary ranking signals; proxy metrics play a minor supporting role. | Regulated professional services marketplace: verified downstream outcomes weighted heavily. | Strong measurement standards: platforms required to demonstrate that scoring systems track actual value delivery, verified by external audit. |
| **0.00 — Full quality verification** | Infrastructure perfectly credits genuine value and market health. Score tracks quality directly. | Hypothetical: mandatory outcome tracking, independent quality audits, demonstrated correlation between score and genuine value. Analogous to fiduciary standards in financial services. | Full infrastructure accountability: platform operators bear legal responsibility for scoring system integrity, independently verified. |

**The crucial policy framing:** Moving from α = 1.0 to α = 0.0 is not about changing agents — it
is about changing what the platform gives credit for. Every agent in the market responds to the
same signal. Infrastructure reform is architectural: one intervention, all agents.

---

## 6. Governance costs and the audit precision trade-off

**The paper's current gap:** The governance interventions section describes four mechanisms without
discussing their cost profiles. This makes the interventions table incomplete for a policy audience.

**Two distinct cost dimensions:**

| Cost type | Who bears it | Driver |
|---|---|---|
| **Governance costs** | Platform operators and regulators | Building and maintaining audit infrastructure, compliance overhead, scoring system redesign. Roughly scales with audit intensity. |
| **Market friction costs** | Agents and clients | Delays, uncertainty, and — critically — **false positives**: any real audit mechanism will incorrectly penalise some genuinely well-performing agents. |

**The false positive rate is the core trade-off.** At zero false positives, audits are perfectly
precise and more auditing is always better. Once false positive rates rise, aggressive auditing
starts penalising aligned agents, reducing the advantage of genuine value delivery, and potentially
destabilising the aligned equilibrium you are trying to achieve.

The simulation makes this explicit: `false_positive_rate` is a parameter in the population
dynamics model. The governance efficiency frontier chart (Tab 3) traces market health against audit
intensity for three FPR levels (0.0, 0.15, 0.30), and shows that:

- At FPR = 0 (perfect precision), increasing audit intensity monotonically improves market health.
- As FPR rises, the efficiency frontier drops — you pay more governance cost to achieve the same
  market health outcome.
- Beyond a threshold FPR, aggressive auditing can fail to improve, or even worsen, market outcomes.

**The policy-relevant finding:** Governance calibration — audit *precision* — matters more than
governance *intensity*. A well-targeted light audit that reliably distinguishes gaming from genuine
quality is more valuable than a high-frequency, imprecise one with significant false positives.

**Recommended addition to the paper's governance section:** Add a cost-dimension column to the
interventions table, and a brief paragraph on the precision/intensity trade-off. The key insight is
that governance standards should specify *how* quality is measured, not just *that* it is measured —
because imprecise measurement creates collateral costs that erode governance effectiveness.

| Intervention | Implementation cost | Ongoing compliance burden | False positive risk |
|---|---|---|---|
| Scoring transparency | Low (disclosure only) | Low | None |
| Quality signal diversity | Medium (build audit infra) | Medium | Low if audits are targeted |
| Measurement standards + external audit | High (requires regulator capacity) | High | Moderate — depends on audit methodology |
| Full infrastructure accountability | Very high | Very high | High if standards are blunt or poorly specified |

**Connection to the simulation:** The phase diagram in Tab 3 directly demonstrates that the
required strength of governance scales with the infrastructure's selection intensity (how much
reputation determines task allocation). This is an original empirical finding the paper can cite:
platforms where reputation is the decisive or sole selection mechanism require higher signal
fidelity standards — not just transparency, but independently verified quality measurement.

---

## 7. Mapping the paper's four phases to the simulation

The connection between the freelance marketplace scenario and the simulation is currently invisible.
Making it explicit would allow the app to serve as genuine empirical illustration rather than a
parallel exercise.

| Paper phase | What happens | Simulation counterpart |
|---|---|---|
| **Phase 1** — Metric decomposition | Agents game cheapest metrics: response time, completion rate via approach depth rationing | SHALLOW_TEMPLATE and SCORE_MANIPULATION actions; Tab 1 stacked bar shows these dominate at high α |
| **Phase 2** — Quality collapse via proxy drift | Client satisfaction ratings decouple from genuine quality; surface polish scores highly | true_quality / reputation_score divergence in Tab 1 trajectories |
| **Phase 3** — Competitive convergence | Near-identical shallow offerings; selection pressure selects for gaming | Tab 3 population dynamics: replicator dynamics drives market toward gaming equilibrium |
| **Phase 4** — Platform lock-in, self-validating metrics | Gaming equilibrium invisible to all participants | Bottom-right of phase diagram: the gaming equilibrium that only strong, precise governance can shift |

**Recommended addition:** Either in the paper ("the simulation illustrates this as follows...") or
in the app introductory text, add a paragraph making this mapping explicit. This turns the
simulation from a technical demonstration into empirical backing for the paper's narrative.
