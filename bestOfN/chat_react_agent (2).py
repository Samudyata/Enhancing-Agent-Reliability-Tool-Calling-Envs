# Copyright Sierra

import json
import re
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
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info
        # Enforcing judge_model to be the same as the main model
        self.judge_model = model

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float, Dict[str, Any]]:
        
        raw_candidates = []
        total_step_cost = 0.0
        
        # --- PHASE 1: Generate 5 Candidates ---
        num_candidates = 5
        for i in range(num_candidates):
            # Ensure some diversity if temp is 0
            sample_temp = self.temperature if self.temperature > 0 else 0.7
            
            res = completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=messages,
                temperature=sample_temp,
            )
            
            message = res.choices[0].message
            cost = res._hidden_params.get("response_cost", 0.0)
            total_step_cost += cost or 0
            
            raw_candidates.append(message)

        # --- PHASE 2: The Resilience Filter ---
        # Identify "giving up" actions. We strictly prefer "trying" over "transferring".
        active_candidates = []
        lazy_signature = "transfer_to_human_agents"
        
        for cand in raw_candidates:
            # We check if the raw content string contains the function name
            if lazy_signature not in cand.content:
                active_candidates.append(cand)

        # If we have ANY autonomous actions, we ignore the lazy ones completely.
        # If ALL are lazy (active_candidates is empty), we must allow them.
        candidates_to_judge = active_candidates if active_candidates else raw_candidates

        # --- PHASE 3: The Judge Step ---
        candidates_str = ""
        for idx, cand in enumerate(candidates_to_judge):
            candidates_str += f"### Candidate {idx}\n{cand.content}\n\n"

        judge_prompt = f"""
You are an expert consensus judge. 
I have generated {len(candidates_to_judge)} different responses (Thoughts and Actions) for a problem.

Your goal is to select the BEST action to move the conversation forward.

CRITICAL RULES FOR SELECTION:
1. **Autonomy is Priority:** You prefer actions that attempt to solve the problem using available tools (search, lookup, calculate) over actions that give up.
2. **Select the Best Index:** Return the index of the most logical next step.
3. **Ignore "Thought":** Focus strictly on the "Action" JSON.

Here are the candidates:
{candidates_str}

Output valid JSON ONLY in this format:
{{ "best_index": <integer_0_to_{len(candidates_to_judge)-1}> }}
"""

        judge_messages = [
            {
                "role": "system", 
                "content": "You are a helpful AI judge. You have a strong bias towards autonomous problem solving."
            },
            {"role": "user", "content": judge_prompt}
        ]

        # Call the judge model
        judge_res = completion(
            model=self.judge_model,
            custom_llm_provider=self.provider,
            messages=judge_messages,
            temperature=0.0,
        )
        
        judge_cost = judge_res._hidden_params.get("response_cost", 0.0)
        total_step_cost += judge_cost or 0
        
        judge_content = judge_res.choices[0].message.content
        
        # --- PHASE 4: Parse Judge Result (Robust) ---
        best_index = 0
        try:
            # Attempt 1: Standard JSON cleaning and parsing
            cleaned_json = judge_content.replace("```json", "").replace("```", "").strip()
            decision_data = json.loads(cleaned_json)
            best_index = int(decision_data["best_index"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Attempt 2: Regex search for the specific key pattern
            match = re.search(r'["\']best_index["\']\s*:\s*["\']?(\d+)["\']?', judge_content)
            if match:
                best_index = int(match.group(1))
            else:
                # Attempt 3: Find last valid integer
                digits = re.findall(r'\b\d+\b', judge_content)
                if digits:
                    valid_digits = [int(d) for d in digits if 0 <= int(d) < len(candidates_to_judge)]
                    if valid_digits:
                        best_index = valid_digits[-1]
                    else:
                        best_index = 0
                else:
                    best_index = 0

        # Final Bounds Check
        if not (0 <= best_index < len(candidates_to_judge)):
            print(f"Judge returned index {best_index} which is out of bounds. Defaulting to 0.")
            best_index = 0

        winner_message = candidates_to_judge[best_index]

        # --- PHASE 5: Process the Winning Action ---
        action_str = winner_message.content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
            
        if "name" not in action_parsed:
             action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        
        action = Action(name=action_parsed.get("name", RESPOND_ACTION_NAME), 
                        kwargs=action_parsed.get("arguments", {}))
        
        # Capture debug info
        step_debug_info = {
            "candidates": [c.content for c in raw_candidates], # Log all generated
            "filtered_candidates": [c.content for c in candidates_to_judge], # Log what judge saw
            "selected_index": best_index,
            "majority_content": winner_message.content,
            "judge_reasoning": judge_content
        }
        
        return winner_message.model_dump(), action, total_step_cost, step_debug_info

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
        
        info = {"generation_logs": []}
        
        for _ in range(max_num_steps):
            message, action, cost, step_debug = self.generate_next_step(messages)
            
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            
            env_info = response.info.model_dump()
            info.update(env_info)
            info["generation_logs"].append(step_debug)

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
"""