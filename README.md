# Enhancing Agent Reliability in Interactive Tool-Calling Environments via Test-Time Scaling

## 📘 Overview
This project investigates **test-time scaling strategies** to improve the reliability of Large Language Model (LLM) agents operating in **interactive tool-calling environments**. Despite recent advances, LLM-based agents frequently fail in real-world scenarios due to domain-rule violations, instruction drift, and multi-step planning breakdowns.

We explore whether **scaling reasoning and interaction depth at inference time**, without increasing model parameters, can mitigate these failures. Using **τ-bench Airline and Retail environments**, our experiments demonstrate that inference-time architectural interventions significantly improve robustness and task success rates.

---

## 🎯 Objectives
- Improve reliability of LLM agents in interactive, tool-based environments.
- Reduce common agent failure modes:
  - Domain rule violations  
  - Context and instruction drift  
  - Planning and execution breakdowns
- Evaluate multiple test-time scaling techniques under a unified benchmark.
- Demonstrate that inference-time reasoning control can outperform parameter scaling.

---

## 🧠 Methods & Architectures

### Best-of-N (BoN)
- Generates *N* independent reasoning trajectories.
- A judge model selects the most consistent or majority-supported output.
- Encourages exploration of diverse reasoning paths but may regress in simpler domains.

### Test-Time Interaction (TTI)
- Extends inference by allowing multiple self-refinement passes.
- The agent is explicitly prompted to reconsider its decision after task completion.
- Evaluated using double-, quadruple-, and sextuple-check strategies.

### Budget Forcing
- Prevents premature action selection by injecting a **“Wait” token** during inference.
- Forces additional reasoning cycles before executing the final action.
- Implemented as an extension of ReAct-style agents.

### Dynamic Budget Steering (DBS)
- Dynamically allocates **Low vs. High thinking budgets** per turn.
- Encourages deeper reasoning only for complex tasks.
- Significantly reduces lazy or insufficient reasoning failures.

### Simulate–Verify–Replan (SVR)
- Introduces a strict verification loop before tool execution.
- Workflow:
  1. Context synthesis  
  2. Simulated outcome prediction  
  3. Verification against constraints and policies  
  4. Final action execution only if verified  
- Effectively filters hallucinations and logical inconsistencies.

---

## 🗂️ Benchmark & Experimental Setup

### Environments
| Environment | Characteristics |
|------------|----------------|
| Airline | Procedural, rule-heavy workflows |
| Retail | Open-ended, multi-step reasoning |

### Models & Stack
- **Agent Model:** Qwen-3-4B  
- **User Model:** Claude-Sonnet-4  
- **Inference Engine:** vLLM  
- **Benchmark:** τ-bench

---

## 📊 Results Summary

### Accuracy Improvements Over Baseline

| Method | Retail | Airline |
|------|--------|---------|
| Best-of-N | +2.6% | −18.0% |
| Test-Time Interaction | Mixed | Stable gains |
| Budget Forcing | −3.5% | +10.0% |
| Dynamic Budget Steering (DBS) | **+33.9%** | +6.0% |
| Simulate–Verify–Replan (SVR) | +20.0% | **+14.0%** |

### Key Findings
- **DBS achieved 43.5% accuracy in Retail**, tripling baseline performance.
- **SVR reached 40.0% accuracy in Airline**, outperforming larger proprietary models.
- Inference-time reasoning control proved more effective than parameter scaling.

---

## 🔍 Analysis & Insights
- Sampling alone cannot compensate for insufficient model reasoning capacity.
- Excessive refinement can destabilize performance in simpler tasks.
- Constraint-heavy domains benefit most from verification-driven reasoning.
- Optimal reasoning depth is **context-dependent**, not fixed.

---

## ✍️ Authors
- Jahnvi Seth  
- Pranesh Somasundar  
- Lekshman Babu Devendra Babu  
- Sravanakumar Sathish  
- Samudyata Sudarshan Jagirdar  

**Mentor:** Amir Saeidi

---

## 📚 References
This project builds upon prior work on τ-bench, test-time scaling, budget steering, and verification-aware planning. Please refer to the project report for the complete list of references.
