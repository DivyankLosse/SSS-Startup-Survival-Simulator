import os
import sys
import json
from openai import OpenAI

from env import StartupEnv
from grader import grade

def run_inference():
    # Read required environment variables
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.", file=sys.stderr)
        hf_token = "dummy_token"

    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token
    )

    tasks = ["survival", "growth", "scaling"]
    valid_actions = [
        "increase_marketing",
        "hire_engineer",
        "improve_product",
        "reduce_costs",
        "pivot_market",
        "raise_funding",
        "do_nothing"
    ]

    for task_name in tasks:
        print(f"[START] Task: {task_name}")
        env = StartupEnv(seed=42)
        env.reset(seed=42)
        done = False
        step_n = 1
        
        while not done:
            current_state = env.state().model_dump()
            
            system_prompt = (
                "You are an AI managing a startup. "
                "Your objective is based on the task. "
                f"Task: {task_name}. "
                f"Valid actions are: {', '.join(valid_actions)}. "
                "Respond entirely with the exact name of your chosen action. No extra text."
            )
            
            user_prompt = f"Current state: {json.dumps(current_state)}"

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                action = response.choices[0].message.content.strip()
                
                # Verify action, fallback to do_nothing if invalid
                if action not in valid_actions:
                    action = "do_nothing"
                    
            except Exception as e:
                # If API call fails, default to do_nothing to not crash the run
                action = "do_nothing"
                
            print(f"[STEP] Step: {step_n}, State: {current_state}, Action: {action}")
            
            try:
                step_result = env.step(action)
                done = step_result["done"]
            except ValueError:
                # e.g., invalid action passed
                done = True

            step_n += 1

        final_state = env.state().model_dump()
        score = grade(task_name, final_state)["score"]
        print(f"[END] Task: {task_name}, Score: {score}")

if __name__ == "__main__":
    run_inference()
