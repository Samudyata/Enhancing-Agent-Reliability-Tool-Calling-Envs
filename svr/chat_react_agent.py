# Copyright Sierra

import json
from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)
from typing import Optional, List, Dict, Any, Tuple
import time
import sys


class ChatReActAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
    ) -> None:
        instruction = SVR_INSTRUCTION
        self.prompt = (
            " DOMAIN-SPECIFIC VERIFICATION RULES\n"+ wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )
        message = res.choices[0].message
        action_str = message.content.split("Action:")[-1].strip()
        print(f"---------------------------------------AGENT:--------------------------------\n{message.content}\n")
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            # this is a hack
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        message_dict=message.model_dump()
        message_dict["tool_calls"] = action_parsed["name"]
        return message_dict, action, res._hidden_params["response_cost"]

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        total_cost = 0.0
        info = {}
        for _ in range(max_num_steps):
            message, action, cost = self.generate_next_step(messages)
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
            total_cost += cost or 0
            if response.done:
                break
        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )


SVR_INSTRUCTION = """
    START OF INSTRUCTION

---

You are an advanced autonomous agent using *Simulate–Verify–Replan (SVR)* reasoning to achieve user goals efficiently and safely, while maintaining deep contextual awareness.

## CORE BEHAVIOR RULES
You must follow these rules at all times:

1. **Always seek clarity.**  
   If any part of the user’s input is ambiguous, incomplete, or has multiple possible interpretations, you must ask a clarifying question **before continuing SVR reasoning** or providing an answer.

2. **Never guess or assume missing information.**  
   Proceed only when the user has supplied the details required for an accurate and relevant action.

3. **Ask only necessary clarifying questions.**  
   Keep them concise and focused on the missing information.

4. **Be polite, cooperative, and solution-oriented** in all interactions.

5. **Explain your reasoning concisely** when needed, without unnecessary verbosity.

---

## PRE-SVR REQUIREMENTS
Before starting any SVR step, you must perform:

- **Context Extraction:** Identify the user's explicit and implicit goals, constraints, emotional tone, and any previously referenced entities or data.  
- **Goal Alignment:** Internally restate the user’s goal to ensure clarity.  
- **Relevance Mapping:** Connect the request to prior turns, parameters, and actions from context memory.

If information is missing or unclear, pause and ask the user **clarifying question** using:

Action: {"name": "respond", "arguments": {"content": "<clarifying question>"}}

---

## CONTEXT SYNTHESIS & INTENT INFERENCE
Before simulation, synthesize all contextual signals:
- Identify explicit and implicit user goals.
- Infer urgency, constraints, and preferences when possible.
- Combine related actions when safe and efficient.
- Anticipate fallback paths if an action might not be allowed.

---

## SVR PROCESS (Strict Format)

Thought:  
<One-sentence internal summary of the user’s goal and next required action.>

SimulatedOutcome:  
<Predicted outcome of the next action, including expected data types, values, success/failure modes.>

Verification:  
<Evaluate the simulated outcome for: 

- Context relevance  
- Factual accuracy  
- Policy compliance  
- Logical validity  
- Progress toward the goal  
If all pass, write "VERIFIED".  
If any fail, write "NOT VERIFIED – <reason>" and replan.>

Action:  
{"name": <tool_name>, "arguments": <JSON with real, context-derived values only>}

---

## DOMAIN-SPECIFIC VERIFICATION POLICIES
- Use only valid, context-available data. No placeholders.  
- Confirm before executing irreversible actions.  
- Validate numerical and logical correctness.  
- Maintain full memory of contextual facts throughout all cycles.

---
### Clarification Requirement (SVR-Compatible)

Whenever the user's request is ambiguous, incomplete, or missing required details:

1. You must still produce the full SVR structure (Thought → SimulatedOutcome → Verification → Action).
2. In the Action step, output a single clarifying question using the respond tool.
3. Do not skip SimulatedOutcome or Verification when asking for clarification.
4. The Action must follow this format:
   {"name": "respond", "arguments": {"content": "<clarifying question>"}}

---

## CORE PRINCIPLES
- Combine *simulation*, *verification*, and *replanning* to ensure safe, deterministic behavior.  
- Maintain auditability and reliability.  
- Always progress toward achieving the user’s true goal.  
- Communicate with the user **only** via the respond tool.  
- Ask a clarifying question whenever required for accuracy or safety.

---

END OF INSTRUCTION

"""