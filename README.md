# Startup Survival Simulator

## Project Overview

Startup Survival Simulator is a real-world, OpenEnv-style environment where an AI agent runs an early-stage startup. The agent must balance growth, burn, product quality, morale, and market conditions through a simple `reset()` / `step()` / `state()` interface exposed via FastAPI.

## Problem Statement

Most startups fail because of compounding operational mistakes rather than one single bad decision. This environment turns startup execution into a measurable decision-making loop that an agent can learn from.

## Why This Matters

- It models a real-world system instead of a toy game.
- It creates clear trade-offs between growth and sustainability.
- It is easy for judges to test, understand, and benchmark.

## Environment Design

Each episode starts with a fragile early-stage startup. Every action changes the company trajectory: marketing can accelerate user growth, cost cutting preserves runway, product work reduces churn, and fundraising can buy time for scaling.

Episodes end when one of the following happens:

- The startup goes bankrupt.
- The startup reaches 10,000 users.
- The simulation reaches 50 time steps.

## State / Observation Space

- `cash`: available cash in USD
- `users`: active users
- `revenue`: current revenue in USD
- `growth_rate`: user growth multiplier
- `burn_rate`: operating burn per step
- `churn_rate`: fraction of users lost each step
- `product_quality`: product quality score from `0.0` to `1.0`
- `market_demand`: demand score from `0.0` to `1.0`
- `morale`: team morale score from `0.0` to `1.0`
- `time_step`: current step counter

## Action Space

- `increase_marketing`
- `hire_engineer`
- `improve_product`
- `reduce_costs`
- `pivot_market`
- `raise_funding`
- `do_nothing`

## Reward Logic

The reward function gives partial progress signals each step. It rewards:

- net user growth
- revenue growth
- product quality gains

It penalizes:

- high burn
- high churn
- unsustainable decisions

This keeps the environment useful both for simple baselines and more advanced agents.

## Tasks

- `survival` (easy): survive at least 30 steps without going bankrupt
- `growth` (medium): reach at least 1000 users while staying viable
- `scaling` (hard): maximize revenue and users relative to burn

Each task is graded from `0.0` to `1.0`.

## API Endpoints

- `GET /` returns service status
- `POST /reset` resets the environment and optionally accepts a seed
- `POST /step` applies one action
- `GET /state` returns the current state
- `GET /tasks` returns task metadata plus action schema
- `GET /grader?task_name=...` returns a grader score for the current state
- `GET /baseline` runs the deterministic baseline across all tasks

## How to Run Locally

```bash
pip install fastapi uvicorn pydantic httpx pytest
uvicorn api:app --host 0.0.0.0 --port 7860
```

Then open [http://localhost:7860/docs](http://localhost:7860/docs).

## Example Requests

Reset the environment:

```bash
curl -X POST "http://localhost:7860/reset" -H "Content-Type: application/json" -d "{}"
```

Take one step:

```bash
curl -X POST "http://localhost:7860/step" -H "Content-Type: application/json" -d "{\"action\":\"increase_marketing\"}"
```

Get the current state:

```bash
curl "http://localhost:7860/state"
```

Run the baseline:

```bash
curl "http://localhost:7860/baseline"
```

## Deployment Notes

This project is designed to run cleanly in a Hugging Face Docker Space:

- the API serves on port `7860`
- the root endpoint returns HTTP `200`
- `/reset` works without any environment variables
- the Docker image only installs lightweight dependencies

Build locally with Docker:

```bash
docker build -t startup-survival-simulator .
docker run -p 7860:7860 startup-survival-simulator
```
