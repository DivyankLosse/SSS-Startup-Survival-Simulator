"""
Inference Script — Startup Survival Simulator
=============================================
MANDATORY environment variables:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN / API_KEY  Your Hugging Face / API key.

STDOUT FORMAT (exact — do not deviate):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
from typing import List, Optional

from openai import OpenAI

from env import StartupEnv
from grader import grade

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL  = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME    = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK     = os.getenv("BENCHMARK",   "startup-survival-simulator")

SUCCESS_SCORE_THRESHOLD = 0.5   # normalised score in [0, 1]

TASKS = ["survival", "growth", "scaling"]

VALID_ACTIONS = [
    "increase_marketing",
    "hire_engineer",
    "improve_product",
    "reduce_costs",
    "pivot_market",
    "raise_funding",
    "do_nothing",
]


# ── Structured log helpers ─────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM action selection ───────────────────────────────────────────────────────
def get_action_from_llm(client: OpenAI, state: dict, task_name: str) -> str:
    system_prompt = (
        "You are an AI CEO managing an early-stage startup. "
        f"Your current objective: {task_name}. "
        f"Choose exactly one action from: {', '.join(VALID_ACTIONS)}. "
        "Reply with only the action name — no explanation, no punctuation."
    )
    user_prompt = f"Current startup state: {json.dumps(state)}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=12,
            stream=False,
        )
        action = (response.choices[0].message.content or "").strip()
        return action if action in VALID_ACTIONS else "do_nothing"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return "do_nothing"


# ── Main episode loop ──────────────────────────────────────────────────────────
def run_inference() -> None:
    if not API_KEY:
        print("Warning: HF_TOKEN / API_KEY is not set.", file=sys.stderr, flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name in TASKS:
        env = StartupEnv(seed=42)
        env.reset(seed=42)

        rewards:     List[float] = []
        steps_taken: int         = 0
        score:       float       = 0.0
        success:     bool        = False
        done:        bool        = False

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            step = 1
            while not done:
                state  = env.state().model_dump()
                action = get_action_from_llm(client, state, task_name)

                error: Optional[str] = None
                try:
                    step_result = env.step(action)
                    reward      = float(step_result.get("reward", 0.0))
                    done        = bool(step_result["done"])
                except ValueError as exc:
                    reward = 0.0
                    done   = True
                    error  = str(exc)

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action, reward=reward, done=done, error=error)

                step += 1

            # Grade against the task objective
            final_state = env.state().model_dump()
            score   = float(grade(task_name, final_state)["score"])
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            print(
                f"[DEBUG] Unexpected error in task {task_name}: {exc}",
                file=sys.stderr,
                flush=True,
            )

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    run_inference()
