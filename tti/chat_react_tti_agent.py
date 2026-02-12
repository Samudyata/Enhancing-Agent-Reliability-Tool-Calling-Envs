# Copyright Sierra (patched)
# ChatReActTTIAgent with robust TTI refinement and R2 included in history.
# Fixed: Trajectory now shows all 4 messages per TTI step

import json
import re
from typing import Optional, List, Dict, Any, Tuple

from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)


def _extract_json_from_action_block(text: str) -> Optional[str]:
    """
    Robustly find the last JSON object after 'Action:'.
    Returns the JSON string if found, otherwise None.
    """
    # Look for the last "Action:" and then capture a {...} block (greedy to end)
    idx = text.rfind("Action:")
    if idx == -1:
        return None
    tail = text[idx + len("Action:") :]
    # Find the first '{' in tail
    start = tail.find("{")
    if start == -1:
        return None
    # Heuristic: find the matching closing brace by simple scanning (handles nested)
    stack = 0
    for i, ch in enumerate(tail[start:], start):
        if ch == "{":
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0:
                return tail[start : i + 1].strip()
    # fallback
    return None


class ChatReActTTIAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.6,
        enable_tti: bool = True,
        tti_rounds: int = 1,
    ) -> None:
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info
        self.enable_tti = enable_tti
        self.tti_rounds = tti_rounds

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
        content = message.content if hasattr(message, "content") else ""
        # Robustly extract JSON after "Action:"
        action_json_str = _extract_json_from_action_block(content)
        if action_json_str:
            try:
                action_parsed = json.loads(action_json_str)
            except json.JSONDecodeError:
                action_parsed = None
        else:
            action_parsed = None

        if not action_parsed:
            # fallback: put whole content as a respond action
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: content.strip()},
            }

        assert "name" in action_parsed and "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])

        # Canonical assistant message dict to return into the convo
        message_dict = {"role": getattr(message, "role", "assistant"), "content": getattr(message, "content", "")}

        # cost guard
        cost = 0.0
        try:
            cost = float(res._hidden_params.get("response_cost", 0.0))
        except Exception:
            cost = 0.0

        return message_dict, action, cost

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
        info: Dict[str, Any] = {}
    
        # Track R1 and R2+ for analysis
        tti_history = []
    
        for step_num in range(max_num_steps):
            # Generate initial action (R1)
            message_r1, action_r1, cost = self.generate_next_step(messages)
            total_cost += (cost or 0.0)
    
            # Store R1
            r1_dict = {"name": action_r1.name, "kwargs": action_r1.kwargs}
    
            # Apply TTI refinement if enabled (R1 -> R2 -> R3 -> ...)
            final_action = action_r1
            action_changed = False
            
            # Track all refinement rounds
            refinement_rounds = []
    
            if self.enable_tti:
                # Add R1 to conversation history (user → assistant R1)
                assistant_r1_msg = {
                    "role": message_r1.get("role", "assistant"), 
                    "content": message_r1.get("content", "")
                }
                messages.append(assistant_r1_msg)
                
                current_action = action_r1
                current_message_content = assistant_r1_msg.get("content", "")
                
                # Multiple rounds of TTI refinement
                for round_num in range(self.tti_rounds):
                    # Create TTI prompt for this round
                    tti_prompt = f"""You just decided to take this action:
    {{"name": "{current_action.name}", "arguments": {json.dumps(current_action.kwargs)}}}
    
    WAIT - Refinement round {round_num + 1}/{self.tti_rounds}. Before executing, let's verify:
    1. Does this action follow the policy and task requirements?
    2. Have you gathered all necessary information?
    3. Is there a more effective action you could take?
    
    Respond with your decision in this exact format:
    Thought:
    <One-line reasoning about whether to keep or change the action>
    Action:
    {{"name": <action_name>, "arguments": <arguments>}}
    
    If you decide to KEEP the original action, just repeat it. If you decide to CHANGE, provide the new action.
    """
    
                    # Add TTI prompt as user message
                    tti_user_msg = {"role": "user", "content": tti_prompt}
                    messages.append(tti_user_msg)
    
                    # Get refinement with thought
                    res = completion(
                        model=self.model,
                        custom_llm_provider=self.provider,
                        messages=messages,
                        temperature=self.temperature,
                    )
    
                    message_refined = res.choices[0].message
                    content = getattr(message_refined, "content", "")
    
                    # Parse the refined action robustly
                    action_json_str = _extract_json_from_action_block(content)
                    if action_json_str:
                        try:
                            action_parsed = json.loads(action_json_str)
                            refined_action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
                        except Exception:
                            refined_action = current_action  # Keep previous if parsing fails
                    else:
                        refined_action = current_action
    
                    # Add refinement cost
                    try:
                        tti_cost = float(res._hidden_params.get("response_cost", 0.0))
                    except Exception:
                        tti_cost = 0.0
                    total_cost += tti_cost
    
                    # Check if action changed in this round
                    refined_dict = {"name": refined_action.name, "kwargs": refined_action.kwargs}
                    current_dict = {"name": current_action.name, "kwargs": current_action.kwargs}
                    round_changed = (refined_dict != current_dict)
    
                    # Store this refinement round
                    refinement_rounds.append({
                        "round": round_num + 1,
                        "input_action": current_dict,
                        "output_action": refined_dict,
                        "changed": round_changed,
                        "message": getattr(message_refined, "content", "")
                    })
    
                    # Add refined message to conversation for next round
                    refined_message = {"role": "assistant", "content": getattr(message_refined, "content", "")}
                    messages.append(refined_message)
                    
                    # Update current action for next round
                    current_action = refined_action
                    current_message_content = getattr(message_refined, "content", "")
    
                # After all rounds, check if final action differs from R1
                final_dict = {"name": current_action.name, "kwargs": current_action.kwargs}
                action_changed = (r1_dict != final_dict)
                
                final_action = current_action
    
                # Store TTI decision + all rounds for debugging
                tti_history.append({
                    "step": step_num,
                    "r1": r1_dict,
                    "final": final_dict,
                    "changed": action_changed,
                    "num_rounds": self.tti_rounds,
                    "rounds": refinement_rounds,
                    "r1_message": assistant_r1_msg.get("content", ""),
                    "final_message": current_message_content,
                })
                
            else:
                # No TTI: just add R1 to conversation
                messages.append(message_r1)
    
            # Execute the (possibly refined) action
            response = env.step(final_action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
    
            if final_action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
    
            # Add observation as user message
            messages.append({"role": "user", "content": obs})
    
            if response.done:
                break
    
        # Add TTI statistics to info
        if self.enable_tti and tti_history:
            total_actions = len(tti_history)
            changed_actions = sum(1 for h in tti_history if h["changed"])
            
            # Additional stats for multi-round TTI
            total_rounds = sum(len(h["rounds"]) for h in tti_history)
            changed_rounds = sum(sum(1 for r in h["rounds"] if r["changed"]) for h in tti_history)
            
            info["tti_stats"] = {
                "total_actions": total_actions,
                "changed_actions": changed_actions,
                "change_rate": changed_actions / total_actions if total_actions > 0 else 0,
                "tti_rounds": self.tti_rounds,
                "total_refinement_rounds": total_rounds,
                "changed_refinement_rounds": changed_rounds,
                "round_change_rate": changed_rounds / total_rounds if total_rounds > 0 else 0,
                "history": tti_history,
            }
    
        # also return total_cost in info for convenience
        info["total_cost"] = total_cost
    
        return SolveResult(messages=messages, reward=reward, info=info)


REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy.
"""


ACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You should not use made-up or placeholder arguments.

The Action will be parsed, so it must be valid JSON.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}
```

Your response can be like this:
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
"""