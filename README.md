# Agentic Digital Twin: Multi-Agent Coordination & Emergent Communication

This repository contains a **digital twin simulation framework** for studying **multi-agent coordination**, **emergent communication**, and **convergence dynamics** in shared-goal environments.

The project focuses on **agent interaction dynamics** rather than full-scale reinforcement learning or natural language processing, providing a controlled setting to analyze how communication, affective state coupling, and coordination signals influence collective behavior.

---

## 🔍 Core Idea

Multiple agents operate in a shared digital twin environment with:
- A common goal location
- Internal affective states (valence, arousal, adaptive weights)
- Optional inter-agent communication

The system studies how:
- Communication accelerates convergence
- Internal agent states synchronize
- Collective behavior emerges without centralized control

---

## 🧠 Key Features

- **Digital Twin Environment**
  - Continuous 2D space
  - Shared global objective
  - Deterministic + stochastic dynamics

- **Multi-Agent System**
  - Configurable number of agents
  - Independent internal states
  - Adaptive movement weighting

- **Emergent Communication**
  - Pairwise agent interactions
  - Communication-dependent convergence effects
  - Communication vs. no-communication comparisons

- **Analysis & Visualization**
  - Convergence plots
  - Cooperation metrics
  - Communication graphs
  - Internal state evolution

---

## 📁 Project Structure

```text
agentic_nlp_rl/
├── agents/                # Agent definitions (baselines, behaviors)
├── envs/
│   └── digital_twin_env.py    # Core digital twin environment
├── experiments/
│   ├── simulate_digital_twin.py
│   ├── compare_communication_convergence.py
│   ├── communication_rate_sweep.py
│   ├── plot_digital_twin_convergence.py
│   ├── plot_agent_internal_convergence.py
│   └── plot_communication_graph.py
├── outputs/               # Generated plots & simulations (gitignored)
├── data/                  # Experimental data (gitignored)
├── logs/                  # Logs (gitignored)
├── .gitignore
└── README.md
