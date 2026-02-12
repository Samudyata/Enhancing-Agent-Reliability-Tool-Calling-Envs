# Enhancing Agent Reliability in Interactive Tool-Calling Environments via Test-Time Scaling

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Qwen](https://img.shields.io/badge/Model-Qwen--3--4B-blueviolet)
![vLLM](https://img.shields.io/badge/Inference-vLLM-ff6f00?logo=lightning&logoColor=white)
![tau-bench](https://img.shields.io/badge/Benchmark-%CF%84--bench-teal)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project investigates **test-time scaling strategies** to improve the reliability of Large Language Model (LLM) agents operating in **interactive tool-calling environments**. Despite recent advances, LLM-based agents frequently fail in real-world scenarios due to domain-rule violations, instruction drift, and multi-step planning breakdowns.

We explore whether **scaling reasoning and interaction depth at inference time** -- without increasing model parameters -- can mitigate these failures. Using the **tau-bench Airline and Retail environments**, our experiments demonstrate that inference-time architectural interventions significantly improve robustness and task success rates.

---

## Objectives

- Improve reliability of LLM agents in interactive, tool-based environments.
- Reduce common agent failure modes:
  - Domain rule violations
  - Context and instruction drift
  - Planning and execution breakdowns
- Evaluate multiple test-time scaling techniques under a unified benchmark.
- Demonstrate that inference-time reasoning control can outperform parameter scaling.

---

## Project Structure

```
.
|-- bestOfN/                          # Best-of-N sampling with judge selection
|   |-- chat_react_agent (2).py       #   Agent: generates N candidates, filters lazy actions, judge picks best
|   |-- Bon Example.png               #   Diagram of the BoN pipeline
|   +-- Bon Example2.png              #   Additional BoN illustration
|
|-- budgetForcing/                    # S1-inspired budget forcing
|   |-- chat_budget_agent.py          #   Agent: appends "Wait" tokens to force reconsideration via vLLM
|   |-- budget_forcing_approach.png   #   Visual overview of the budget forcing approach
|   |-- exampleBF.png                 #   Example trace showing forced reconsideration
|   |-- airline_budget_exclusive_wins_budget_forcing.json   # Airline-domain exclusive wins
|   +-- retail_budget_exclusive_wins_budget_forcing.json    # Retail-domain exclusive wins
|
|-- dbs/                              # Dynamic Budget Steering (DBS)
|   |-- chat_react_agent.py           #   Agent: classifies task complexity (LOW/MEDIUM/HIGH), adjusts reasoning depth
|   +-- example.jpeg                  #   Example of budget-guided reasoning trace
|
|-- svr/                              # Simulate-Verify-Replan (SVR)
|   |-- chat_react_agent.py           #   Agent: simulates outcomes, verifies against constraints, replans on failure
|   +-- airline.json                  #   Airline-domain results
|
|-- tti/                              # Test-Time Interaction (TTI)
|   |-- chat_react_tti_agent.py       #   Agent: multi-round self-refinement before executing actions
|   |-- results.png                   #   TTI results visualization
|   +-- TTI_Examples.pdf              #   Detailed TTI example traces
|
|-- Final Report.pdf                  # Full project report
|-- NLP_FinalPoster.pdf               # Conference-style poster
|-- PromptOps - pre presentation.pptx # Presentation slides
|-- README.md                         # This file
|-- requirements.txt                  # Python dependencies
+-- .gitignore                        # Git ignore rules
```

---

## Methods & Architectures

### Best-of-N (BoN)

Generates **N** independent reasoning trajectories per step. A resilience filter removes lazy actions (e.g., premature transfers to human agents), and a judge model selects the most logical candidate. This encourages exploration of diverse reasoning paths but may regress in simpler domains.

### Test-Time Interaction (TTI)

Extends inference by allowing multiple self-refinement passes. After the agent proposes an initial action, it is explicitly prompted to reconsider its decision through configurable refinement rounds (double-check, quadruple-check, sextuple-check strategies).

### Budget Forcing

Prevents premature action selection by injecting a **"Wait" token** during inference, inspired by the S1 approach. This forces additional reasoning cycles before executing the final action and is implemented via the vLLM OpenAI-compatible API.

### Dynamic Budget Steering (DBS)

Dynamically allocates **Low / Medium / High thinking budgets** per turn based on task complexity estimation. The agent classifies each step complexity, then adjusts its reasoning depth accordingly -- encouraging deeper analysis only when warranted. This significantly reduces lazy or insufficient reasoning failures.

### Simulate-Verify-Replan (SVR)

Introduces a strict verification loop before tool execution:

1. **Context Synthesis** -- identify goals, constraints, and intent
2. **Simulated Outcome** -- predict the result of the proposed action
3. **Verification** -- check against policies, logic, and factual accuracy
4. **Action Execution** -- proceed only if verified; otherwise replan

This effectively filters hallucinations and logical inconsistencies.

---

## Getting Started

### Prerequisites

| Requirement | Version / Notes |
|---|---|
| Python | >= 3.10 |
| CUDA | GPU with >= 16 GB VRAM recommended |
| vLLM | For local model serving |
| tau-bench | Sierra agent benchmark |

### 1. Clone the repository

```bash
git clone https://github.com/Samudyata/Enhancing-Agent-Reliability-Tool-Calling-Envs.git
cd Enhancing-Agent-Reliability-Tool-Calling-Envs
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start vLLM

```bash
pip install vllm
```

Serve the Qwen model locally:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8005 \
    --dtype auto
```

### 4. Install tau-bench

```bash
git clone https://github.com/sierra-research/tau-bench.git
cd tau-bench
pip install -e .
```

### 5. Set API keys

Some methods use LiteLLM, which routes to different providers. Set keys as needed:

```bash
export ANTHROPIC_API_KEY="sk-..."   # For Claude (user model)
export OPENAI_API_KEY="EMPTY"       # For local vLLM
```

---

## Running Experiments

All agents extend the `tau_bench.agents.base.Agent` class and are evaluated using the tau-bench harness. Below are example commands for each method.

### Best-of-N (BoN)

```bash
python -m tau_bench.run \
    --agent-strategy chat_react \
    --agent-module bestOfN.chat_react_agent \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --env airline \
    --num-trials 5
```

### Test-Time Interaction (TTI)

```bash
python -m tau_bench.run \
    --agent-strategy chat_react \
    --agent-module tti.chat_react_tti_agent \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --env retail \
    --num-trials 5
```

### Budget Forcing

```bash
python -m tau_bench.run \
    --agent-strategy chat_react \
    --agent-module budgetForcing.chat_budget_agent \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --env airline \
    --num-trials 5
```

### Dynamic Budget Steering (DBS)

```bash
python -m tau_bench.run \
    --agent-strategy chat_react \
    --agent-module dbs.chat_react_agent \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --env retail \
    --num-trials 5
```

### Simulate-Verify-Replan (SVR)

```bash
python -m tau_bench.run \
    --agent-strategy chat_react \
    --agent-module svr.chat_react_agent \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --env airline \
    --num-trials 5
```

> **Note:** Adjust `--env` to `airline` or `retail` and `--num-trials` as needed. The `--model` flag should match the model being served by vLLM.

---

## Benchmark & Experimental Setup

### Environments

| Environment | Characteristics |
|---|---|
| **Airline** | Procedural, rule-heavy workflows (booking changes, cancellations, policy enforcement) |
| **Retail** | Open-ended, multi-step reasoning (product lookup, order management, returns) |

### Models & Stack

| Component | Choice |
|---|---|
| Agent Model | Qwen-3-4B |
| User Model | Claude Sonnet 4 |
| Inference Engine | vLLM |
| Benchmark | tau-bench |

---

## Results Summary

### Accuracy Improvements Over Baseline

| Method | Retail | Airline |
|---|:---:|:---:|
| Best-of-N | +2.6% | -18.0% |
| Test-Time Interaction | Mixed | Stable gains |
| Budget Forcing | -3.5% | +10.0% |
| **Dynamic Budget Steering (DBS)** | **+33.9%** | +6.0% |
| **Simulate-Verify-Replan (SVR)** | +20.0% | **+14.0%** |

### Key Findings

- **DBS achieved 43.5% accuracy in Retail**, tripling baseline performance.
- **SVR reached 40.0% accuracy in Airline**, outperforming larger proprietary models.
- Inference-time reasoning control proved more effective than parameter scaling.

---

## Analysis & Insights

- Sampling alone cannot compensate for insufficient model reasoning capacity.
- Excessive refinement can destabilize performance in simpler tasks.
- Constraint-heavy domains benefit most from verification-driven reasoning.
- Optimal reasoning depth is **context-dependent**, not fixed.

---

## Authors

- Jahnvi Seth
- Pranesh Somasundar
- Lekshman Babu Devendra Babu
- Sravanakumar Sathish
- Samudyata Sudarshan Jagirdar

**Mentor:** Amir Saeidi

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{seth2025enhancing,
  title     = {Enhancing Agent Reliability in Interactive Tool-Calling Environments via Test-Time Scaling},
  author    = {Seth, Jahnvi and Somasundar, Pranesh and Devendra Babu, Lekshman Babu and Sathish, Sravanakumar and Jagirdar, Samudyata Sudarshan},
  year      = {2025},
  note      = {Course project report}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## References

This project builds upon prior work on tau-bench, test-time scaling, budget steering, and verification-aware planning. Please refer to `Final Report.pdf` for the complete list of references.
