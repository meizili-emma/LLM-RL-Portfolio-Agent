# Large Language Model–Driven Reinforcement Learning for Weekly Portfolio Allocation

This repository contains the code and documentation for my MPhil thesis project at the Department of Computer Science and Engineering, The Hong Kong University of Science and Technology.

The project studies how large language models (LLMs) can be connected with reinforcement learning (RL) for weekly portfolio allocation. In particular, it asks whether LLM-derived signals from financial text can add value beyond strong price-based and classical portfolio baselines.

This repository is mainly a thesis research artifact and technical portfolio project. It shares the system design, implementation structure, and empirical observations. It does not include raw financial datasets, processed feature panels, API keys, or generated experiment logs.

---

## Thesis Materials

- [MPhil Thesis](docs/MeiziLi_FinalThesis_MPhil_CSE_HKUST.pdf)
- [Defense Slides](docs/Defense.pdf)

For full methodological details, experiment tables, and discussion, please refer to the thesis.

---

## Project Summary

Portfolio allocation is a sequential decision problem: at each rebalancing date, an agent chooses how to distribute capital across a set of assets while balancing return, risk, transaction costs, and changing market regimes.

This project instantiates the problem as a weekly allocation task over 20 liquid U.S. large-cap equities from 2019 to 2025. The system combines:

- a weekly portfolio RL environment;
- a PPO-based allocation agent;
- technical and market features;
- structured LLM signals from earnings calls, SEC filings, news, and technical commentary;
- risk-aware reward shaping;
- evaluation of performance, risk, turnover, and portfolio concentration.

The goal is not to propose a live trading system. The goal is to study how LLM-derived information behaves when integrated into a controlled RL portfolio framework.

---

## Research Questions

The thesis focuses on three questions.

### RQ1: LLM Signals as Observation Features

Do LLM-derived directional signals from financial text improve a weekly RL allocator compared with a price-only PPO agent?

### RQ2: LLM-Based Risk Shaping

Can LLM-derived risk scores help shape the RL reward and encourage more downside-aware portfolio behavior?

### RQ3: Training Mechanisms

Does curriculum-style feature introduction improve stability or robustness compared with all-in-one multimodal training?

---

## System Overview

The system follows this high-level pipeline:

```text
Raw market and text data
        |
        v
Data cleaning and weekly alignment
        |
        v
LLM analyst pipeline
        |
        |-- SEC analyst
        |-- Earnings-call analyst
        |-- News analyst
        |-- Technical analyst
        |-- Tier-2 senior analyst
        |
        v
Structured signals and risk scores
        |
        v
Weekly portfolio RL environment
        |
        v
PPO training and evaluation
        |
        v
Risk, return, turnover, and behavior analysis
```

The LLMs do not directly make trading decisions. They act as structured information extractors. The RL agent remains responsible for portfolio allocation and policy learning.

---

## Main Method

### 1. Weekly Portfolio Environment

The environment formulates portfolio allocation as a weekly Markov decision process.

At each week, the agent observes portfolio state and selected market/LLM features, then outputs a target portfolio weight vector. The environment applies transaction costs, updates portfolio value, and returns a scalar reward.

Key features:

- target-weight portfolio actions;
- weekly rebalancing;
- proportional transaction costs;
- cash and asset-weight tracking;
- optional LLM features;
- risk-aware reward terms;
- per-step and per-episode logging.

### 2. LLM Analyst Pipeline

The LLM pipeline converts heterogeneous financial text into structured records for downstream RL.

The main analyst channels are:

- **SEC analyst**: long-horizon structural information from filings;
- **Earnings-call analyst**: management outlook and quarterly operating updates;
- **News analyst**: event-driven weekly updates;
- **Technical analyst**: price/volume-based market commentary;
- **Tier-2 analyst**: synthesis across Tier-1 analyst outputs.

The structured output includes fields such as:

```text
signal       directional score
risk_score   downside risk score
confidence   reliability estimate
rationale    short explanation
```

### 3. PPO Portfolio Agent

The RL agent uses Proximal Policy Optimization (PPO) with a custom portfolio feature extractor. The experiments compare price-only PPO against LLM-enhanced variants.

LLM information is integrated in two main ways:

- as additional observation features;
- as exogenous risk terms in the reward.

### 4. Evaluation

The evaluation considers both aggregate performance and trading behavior.

Metrics include:

- net asset value;
- annualized NAV;
- Sharpe ratio;
- maximum drawdown;
- turnover;
- transaction cost;
- portfolio concentration;
- train/test generalization;
- allocation behavior over time.

---

## Key Observations

The empirical findings are intentionally mixed.

1. **Price-only PPO is a strong baseline.**  
   A tuned price-only PPO agent can match or exceed classical equal-weight and minimum-variance baselines in the tested setting.

2. **LLM features change behavior but do not robustly dominate.**  
   Adding LLM-derived signals affects trading style, turnover, concentration, and sector tilts, but does not consistently outperform the price-only PPO backbone.

3. **LLM risk shaping has limited but informative effects.**  
   LLM-derived risk penalties can encourage more defensive behavior in some configurations, but the gains are modest and fragile.

4. **Curriculum training is not automatically better.**  
   Staged feature introduction can be useful for analysis, but it does not reliably dominate all-in-one multimodal training in this finite weekly-data setting.

5. **The main contribution is diagnostic and methodological.**  
   The project shows how to build and evaluate an LLM-to-RL portfolio pipeline with structured signals, controlled ablations, and behavior-aware analysis.

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── .env_sample
├── .gitignore
├── environment.yml
├── pyproject.toml
├── run_ollama.sh
├── docs/
│   ├── Defense.pdf
│   └── MeiziLi_FinalThesis_MPhil_CSE_HKUST.pdf
├── data/
│   ├── raw/
│   ├── processed/
│   ├── rl_runs/
│   ├── logs/
│   └── debug/
└── src/
    ├── analysis/
    ├── data_pipeline/
    ├── llm_agents/
    └── rl_agents/
```

The `data/` directories are placeholders only. The repository does not include raw financial data, processed feature panels, generated LLM outputs, or experiment results.

---

## Code Organization

### `src/data_pipeline/`

Data ingestion, preprocessing, technical indicator construction, weekly calendar alignment, and feature panel preparation.

### `src/llm_agents/`

LLM analyst schemas, prompts, structured extraction utilities, Tier-1 analyst modules, and Tier-2 synthesis.

### `src/rl_agents/`

Portfolio environment, PPO training scripts, feature extractor, configuration files, debugging utilities, and batch-run support.

### `src/analysis/`

Experiment log loading, metrics, aggregation, plotting, and per-run reporting utilities.

---

## Environment Setup

The repository includes an environment file for recreating the main Python environment:

```bash
conda env create -f environment.yml
conda activate portfolio_agent
pip install -e .
```

Create a local `.env` file from the sample:

```bash
cp .env_sample .env
```

Then fill in local API keys and endpoints as needed.

The `.env` file is ignored by Git and should not be committed.

---

## Reproducibility Note

This repository does not provide full end-to-end reproduction out of the box because the underlying financial datasets, processed feature panels, API keys, and experiment outputs are not included.

The code is provided to document the research implementation and system design. To reproduce the full experiments, users need to prepare compatible local market data, textual data, LLM outputs, and processed feature panels following the structure used by the pipeline.

---

## Limitations

Important limitations are discussed in detail in the thesis. The main ones are:

- limited weekly sample size;
- non-stationary market regimes from 2019 to 2025;
- public-information efficiency in liquid U.S. equities;
- possible LLM pretraining contamination despite timestamp-based document alignment;
- sensitivity to data coverage, prompt design, and PPO hyperparameters;
- lack of claim to live trading profitability.

This repository is for academic research and technical demonstration only.

---

## Contact

For questions, collaboration, or research discussion, please feel free to contact me.

**Meizi Li**  
MPhil in Computer Science and Engineering  
The Hong Kong University of Science and Technology  

- Email: mlicr@connect.ust.hk
- GitHub: https://github.com/meizili-emma

I am interested in research collaborations related to large language model agents, reinforcement learning, financial decision systems, and long-horizon sequential decision-making.

---

## Citation

If you use or refer to this project, please cite the thesis:

```bibtex
@mastersthesis{li2026llmrlportfolio,
  title  = {Large Language Model Driven Reinforcement Learning for Portfolio Allocation},
  author = {Li, Meizi},
  school = {The Hong Kong University of Science and Technology},
  year   = {2026}
}
```

---

## License

This code is released under the MIT License.

The repository does not include raw financial datasets, API keys, generated experiment outputs, or investment recommendations.

---

## Disclaimer

This repository is for academic research and technical demonstration only. It does not provide financial advice, investment recommendations, or live trading signals. Past backtest performance does not imply future performance.