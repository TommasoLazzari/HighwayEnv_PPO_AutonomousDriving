# Deep Reinforcement Learning for Autonomous Driving on Multilane Highway

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.1.1-green.svg)](https://gymnasium.farama.org/)
[![Highway-Env](https://img.shields.io/badge/Highway--Env-v1.10.2-orange.svg)](https://github.com/Farama-Foundation/HighwayEnv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

#### Author: Tommaso Lazzari
*MSc in Data Science — University of Padua*  
📄 **Full Technical Paper:** [`Report.pdf`](./Report.pdf)

---

## Project Overview

This project investigates the application of **Deep Reinforcement Learning (DRL)** to autonomous vehicle control in high-density, multilane highway traffic environments using the [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) simulation benchmark.

The core objective is to safely navigate an ego-vehicle at high speed through congested traffic while avoiding collisions, adhering to rightmost-lane driving norms, and recovering from dynamic traffic bottlenecks.

### Key Research Questions
1. **DRL vs. Heuristic Benchmark**: Can a self-learning agent trained via **Proximal Policy Optimization (PPO)** outperform an educated, human-engineered deterministic baseline policy?
2. **Safety Regularization & Reward Shaping**: How does introducing an explicit terminal collision penalty (*crash malus*) alter the policy's risk-reward trade-off between driving speed and vehicle survival?
3. **Environment Scalability**: How do baseline and RL policies scale when transitioning across highway environments with varying spatial constraints (3, 4, and 5 lanes)?

---

## Key Experimental Results

The policies were tested across **3-lane**, **4-lane**, and **5-lane** highway configurations. Evaluation performance represents the average across **10 deterministic test runs** using greedy action selection ($\text{argmax}$).

### Quantitative Evaluation Summary

| Lanes | Policy Configuration | Mean Return | Crash Rate (%) | Return Variance ($s^2$) |
| :---: | :--- | :---: | :---: | :---: |
| **3** | **Safety-First Heuristic Baseline** | 27.358 | 10.0% | 40.9 |
| **3** | **PPO (Default Reward)** | 26.434 | 10.0% | 17.9 |
| **3** | **PPO (With Crash Malus)** | **28.627** | **0.0%** | **1.5** |
| **4** | **PPO (Default Reward)** | 27.939 | **0.0%** | 1.4 |
| **4** | **PPO (With Crash Malus)** | **28.294** | **0.0%** | **0.5** |
| **5** | **PPO (Default Reward)** | 28.028 | **0.0%** | 1.1 |
| **5** | **PPO (With Crash Malus)** | 28.028 | **0.0%** | 1.2 |

### Main Takeaways
- **Zero-Collision Driving**: Introducing the $-5.0$ crash malus eliminated collisions entirely across all lane configurations ($0.0\%$ evaluation crash rate).
- **Drastic Variance Reduction**: In 3 lanes, the crash-malus PPO agent achieved an evaluation return variance **over 25 times lower** than the baseline policy ($s^2 = 1.5$ vs. $40.9$), proving exceptional trajectory consistency and robustness against critical traffic bottlenecks.
- **Spatial Freedom vs. Density**: In 4- and 5-lane environments, additional lateral space acts as a natural safety buffer, reducing overall traffic density and accelerating convergence.

---

## Directory Structure

```text
autonomous_driving/
│
├── Report.pdf                     # Complete 7-page technical research report
│
├── ppo_agent.py                   # PPO architecture (ActorCritic, PPOBuffer, GAE, PPOAgent)
├── training.py                    # Training pipeline with LR annealing and reward shaping
├── evaluate.py                    # Deterministic evaluation script with GUI rendering
├── your_baseline.py               # Deterministic rule-based baseline policy benchmark
├── manual_control.py              # Interactive manual keyboard driving interface
│
├── weights/                       # Pretrained PyTorch model weights (.pt)
│   ├── ppo_agent_final_3lanes.pt
│   ├── ppo_agent_final_3lanes_crashmalus.pt
│   ├── ppo_agent_final_4lanes.pt
│   ├── ppo_agent_final_4lanes_crashmalus.pt
│   ├── ppo_agent_final_5lanes.pt
│   └── ppo_agent_final_5lanes_crashmalus.pt
│
├── results/                       # Training logs and performance histories (.json)
│   ├── training_log_3lanes.json
│   ├── training_log_3lanes_crashmalus.json
│   ├── training_log_4lanes.json
│   ├── training_log_4lanes_crashmalus.json
│   ├── training_log_5lanes.json
│   └── training_log_5lanes_crashmalus.json
│
├── baseline_results.json          # 10-episode evaluation metrics for baseline policy
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project documentation
```

---

## Methodology & Architecture

### 1. Environment & State Space
- **Simulator**: `highway-fast-v0` (for accelerated training) and `highway-v0` (for evaluation rendering).
- **Observation Space**: A $5 \times 5$ matrix capturing the ego-vehicle and 4 neighboring vehicles, flattened into a 25-dimensional state vector:
  $\mathbf{s} = \left[ \text{presence}_i,\, x_i,\, y_i,\, v_{x,i},\, v_{y,i} \right]^T \quad \text{for } i \in \{1, \dots, 5\}$
- **Action Space**: 5 discrete meta-actions:
  `LANE_LEFT` (0), `IDLE` (1), `LANE_RIGHT` (2), `FASTER` (3), `SLOWER` (4).

### 2. Safety-First Deterministic Baseline
A rule-based policy emulating cautious human decision-making:
- Accelerate when the lane ahead is clear.
- When an obstacle is within `CAUTION_FRONT` ($0.30$), check adjacent lanes using lateral proximity heuristic `LANE_THRESHOLD` ($0.25$), `SAFE_FRONT` ($0.30$), and `SAFE_REAR` ($0.10$).
- Prioritize changing to the right lane (higher reward), then left lane; if neither is safe, decelerate.

### 3. Proximal Policy Optimization (PPO)
- **Actor-Critic Architecture**:
  - **Actor**: MLP with 2 hidden layers (256 units, ReLU activations) $\to$ 5 action logits, initialized orthogonally ($\sigma = 0.01$).
  - **Critic**: MLP with 2 hidden layers (256 units, ReLU activations) $\to$ scalar state-value $V(s)$, initialized orthogonally ($\sigma = 1.0$).
- **Advantage Estimation**: Generalized Advantage Estimation (GAE) with $\gamma = 0.99$ and $\lambda = 0.95$.
- **Clipped Objective**: Policy ratio clipping with $\epsilon = 0.20$.
- **Optimizer**: Adam with learning rate annealing from $3 \times 10^{-4}$ down to $5 \times 10^{-5}$ over 60,000 steps.

### 4. Reward Shaping (Crash Malus)
- **Default Highway-Env Reward**:
  $$R(s, a) = w_v \frac{v - v_{\min}}{v_{\max} - v_{\min}} + w_l d_{\text{right}} - w_c \mathbb{I}(\text{collision})$$
  with $w_v = 0.4$, $w_l = 0.1$, $w_c = 1.0$.
- **Modified Reward (Safety Regularized)**:
  $$R_{\text{modified}}(s, a) = R(s, a) - 5.0 \cdot \mathbb{I}(\text{crash})$$
  Adding an extra terminal penalty of $-5.0$ heavily penalizes catastrophic decisions and prioritizes vehicle survival over marginal speed gains.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/TommasoLazzari/<REPOSITORY_NAME>.git
cd <REPOSITORY_NAME>
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Evaluate Pre-trained PPO Agent (with Visual Rendering)
Run the trained PPO agent with greedy action selection in the human rendering window:
```bash
python evaluate.py
```
> *Tip: You can switch model weights (e.g., 3, 4, or 5 lanes with or without crash malus) directly in `evaluate.py` by pointing `agent.load(...)` to any checkpoint in `weights/`.*

### 2. Run the Heuristic Baseline Benchmark
Evaluate the handcrafted safety-first baseline policy and record 10-episode metrics:
```bash
python your_baseline.py
```

### 3. Train a PPO Agent from Scratch
Train a new PPO policy using either the default or crash-malus reward function:
```bash
python training.py
```
> *Configure lane count (`lanes_count: 3, 4, 5`) and reward shaping in `training.py` (lines 27 & 85).*

### 4. Interactive Manual Driving (Human Control)
Test human performance using the arrow keys to control the ego-vehicle:
```bash
python manual_control.py
```

---

## References

1. **Leurent, E.** (2018). *An Environment for Autonomous Driving on Highway*. [GitHub Repository](https://github.com/Farama-Foundation/HighwayEnv).
2. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). *Proximal Policy Optimization Algorithms*. arXiv preprint [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).

---

## Author & Contact

- **Tommaso Lazzari** — MSc Data Science Student, University of Padua
- **LinkedIn:** [linkedin.com/in/tommaso-lazzari-datascience](https://www.linkedin.com/in/tommaso-lazzari-datascience/)
- **GitHub:** [@TommasoLazzari](https://github.com/TommasoLazzari)
- **Email:** [tommaso.lazzari02@gmail.com](mailto:tommaso.lazzari02@gmail.com)
