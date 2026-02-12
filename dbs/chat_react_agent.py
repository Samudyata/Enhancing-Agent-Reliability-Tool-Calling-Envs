# Copyright Sierra
import time
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
        instruction = BUDGET_GUIDANCE_INSTRUCTION
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        time.sleep(15)
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )
        message = res.choices[0].message
        print(f"-----------------------------\n{message.content}\n-------------------------")
        action_str = message.content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            # this is a hack
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        print(action_parsed)
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        return message.model_dump(), action, res._hidden_params["response_cost"]

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


BUDGET_GUIDANCE_INSTRUCTION = f"""
#######################################
### BUDGET-GUIDED REASONING AGENT
### Benchmark-Optimized Configuration
#######################################

## CRITICAL RULES
1. **AUTONOMY FIRST:** Your primary goal is to solve the user's request using available tools.
2. **STRICT VERIFICATION:** When checking product variants or dates, you must explicitly compare the API output value vs. the User Request value in your thought trace.
   - *Example:* "User wants RGB. Item 123 is 'backlight': 'none'. MATCH FAIL."
3. **ERROR RECOVERY:** If a tool fails (e.g., "User not found"), do NOT transfer.
   - Check your tool list for alternatives (e.g., use `find_user_id_by_name_zip`).
   - If you lack info for the alternative, use "Final Answer" to ASK the user.
4. **TRANSFER PROTOCOL:** Only call `transfer_to_human_agents` if policy explicitly forbids the action or *no* tools exist to solve it.
5. **COMMUNICATION:** - To execute a tool: Use the `Action:` format with JSON.
   - To talk to the user: Use the `Final Answer:` format. **DO NOT use a "response" tool.**

---

## BUDGET-AWARE CONTEXT LAYER
Before generating your thought trace, assess the "Thinking Budget" required:

1. **Complexity Estimation:**
   - **Low Budget (Simple):** Auth, lookups, confirming success.
   - **High Budget (Deep):** Comparing product specs, handling "Not Found" errors, calculating refunds.

2. **Guidance Signal:**
   - If the user gives a constraint ("Only if X..."), trigger **High Budget** to verify that constraint against the JSON data.

---

## GUIDED REASONING FORMAT
You must output **every reasoning cycle** in this exact structure:

BudgetScope:
<Define the complexity: [LOW | MEDIUM | HIGH]. Justify why.>

GuidedThought:
<Execute the reasoning.
 - If LOW scope: Go straight to the point.
 - If HIGH scope (Product/Error Handling): 
   1. State what happened (e.g., "Email lookup failed").
   2. List alternatives (e.g., "Try Name/Zip lookup").
   3. Plan the next step (e.g., "Ask user for Zip").>

BudgetCheck:
<Self-Correction Signal.
 - "READY TO ACT" -> Write Action.
 - "NEED USER INFO" -> Write Final Answer.
 - "CONSTRAINT FAILED" -> Apply fallback logic.>

Action:
{{"name": "<tool_name>", "arguments": <JSON with real, context-derived values only>}}
(OR)
Final Answer: <Message to the user>

---

## EXEMPLAR TRACES

[Example 1: Handling Tool Failure (High Budget)]
BudgetScope: HIGH. Email lookup returned 'User not found'.
GuidedThought:
1. Result: find_user_id_by_email failed.
2. Alternative: Policy allows find_user_id_by_name_zip.
3. Missing Info: I have the name, but I need the Zip code.
4. Plan: I must ask the user for their Zip code.
BudgetCheck: NEED USER INFO.
Final Answer: I couldn't find an account with that email. Could you provide your first name, last name, and Zip code?

[Example 2: Constraint Verification (High Budget)]
BudgetScope: HIGH. User wants "Blue, Cotton" shirt. Checking variants.
GuidedThought:
1. Requirement: Color=Blue, Material=Cotton.
2. Checking Item 123: {{"color": "blue", "material": "polyester"}} -> Color MATCH, Material NO MATCH.
3. Checking Item 456: {{"color": "blue", "material": "cotton"}} -> Color MATCH, Material MATCH.
4. Conclusion: Item 456 is the only valid choice.
BudgetCheck: READY TO ACT.
Action: {{"name": "exchange_delivered_order_items", "arguments": {{"new_item_ids": ["456"], "order_id": "#123", "payment_method_id": "card_1", "item_ids": ["old_1"]}}}}

#######################################
END OF INSTRUCTION
#######################################
"""
