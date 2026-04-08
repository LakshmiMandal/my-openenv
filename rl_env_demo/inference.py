"""
Inference Script for Smart Home Energy Management Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import textwrap
import numpy as np
from typing import List, Optional, Dict, Any
import asyncio

from openai import OpenAI

# Import from rl_env_demo
from models import SmartHomeObservation
from tasks import get_grader, get_task_config, ALL_TASKS
from client import SmartHomeEnv
from models import SmartHomeAction

# Environment variables
# API_KEY="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"
IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "cost_minimization")
BENCHMARK = "smart_home_energy"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "rl_env_demo-env:latest")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

# Task configuration
MAX_STEPS = 96  # 24 hours in 15-minute intervals
TEMPERATURE = 0.7
MAX_TOKENS = 150

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a smart home energy management system.
    Your goal is to optimize energy usage by controlling:
    - HVAC (heating/cooling): 0=off, 1=heating, 2=cooling
    - Battery: 0=idle, 1=charge, 2=discharge
    - Appliances: 0=minimal, 1=normal, 2=high usage
    
    Actions are encoded as: action = hvac * 9 + battery * 3 + appliances
    Valid actions are integers from 0 to 26.
    
    Consider:
    - Time of day and electricity pricing
    - Indoor/outdoor temperature
    - Solar generation availability
    - Battery charge level
    - Home occupancy
    
    Respond with only the action number (0-26).
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log each step."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, observation: SmartHomeObservation, last_reward: float, info: dict) -> str:
    """Build prompt for the model based on current observation."""
    # Denormalize key features for better understanding
    hour = int(observation.hour_of_day * 23)
    indoor_temp = observation.indoor_temp * 15.0 + 15.0
    outdoor_temp = observation.outdoor_temp * 65.0 - 20.0
    solar_gen = observation.solar_generation * 10.0
    battery = observation.battery_charge * 100.0
    price = observation.electricity_price * 0.45 + 0.05
    occupancy = "home" if observation.occupancy > 0.5 else "away"
    
    return textwrap.dedent(
        f"""
        Step: {step}/{MAX_STEPS}
        Time: Hour {hour}:00
        Indoor Temperature: {indoor_temp:.1f}°C
        Outdoor Temperature: {outdoor_temp:.1f}°C
        Solar Generation: {solar_gen:.1f} kW
        Battery Charge: {battery:.1f}%
        Electricity Price: ${price:.2f}/kWh
        Occupancy: {occupancy}
        Last Reward: {last_reward:.2f}
        
        Choose action (0-26):
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, observation: SmartHomeObservation, last_reward: float, info: dict) -> int:
    """Get action from LLM."""
    user_prompt = build_user_prompt(step, observation, last_reward, info)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Extract action number from response
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            action = int(numbers[0])
            if 0 <= action <= 26:
                return action
        
        # Default to safe action if parsing fails
        return 0  # All systems off
        
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 0  # Default safe action


async def main() -> None:
    """Main inference loop."""
    # Validate task
    if TASK_NAME not in [task.task_id for task in ALL_TASKS]:
        print(f"[ERROR] Unknown task: {TASK_NAME}", flush=True)
        print(f"[ERROR] Available tasks: {[task.task_id for task in ALL_TASKS]}", flush=True)
        sys.exit(1)
    
    # Initialize OpenAI client
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY environment variable not set", flush=True)
        sys.exit(1)
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Get task configuration
    task_config = get_task_config(TASK_NAME)
    grader = get_grader(TASK_NAME)
    
    # Initialize environment client
    try:
        if LOCAL_IMAGE_NAME:
            print(
                "[DEBUG] LOCAL_IMAGE_NAME is set, but from_docker_image() returned an async object in this runtime; using SERVER_URL instead.",
                flush=True,
            )
        env = SmartHomeEnv(base_url=SERVER_URL)
    except Exception as e:
        print(f"[ERROR] Failed to connect to environment: {e}", flush=True)
        sys.exit(1)
    
    # Episode tracking
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    episode_info = {
        'total_cost': 0.0,
        'comfort_violations': 0,
        'total_solar_used': 0.0,
        'steps_completed': 0,
        'battery_cycles': 0,
        'avg_temp_deviation': 0.0
    }
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Reset environment
        result = await env.reset()
        observation = result.observation
        last_reward = 0.0
        done = False
        
        temp_deviations = []
        last_battery_action = 0
        
        # Episode loop
        for step in range(1, MAX_STEPS + 1):
            if done:
                break
            
            # Get action from model
            action_id = get_model_action(client, step, observation, last_reward, episode_info)
            
            # Take step in environment
            try:
                action = SmartHomeAction(action_id=action_id)
                result = await env.step(action)
                observation = result.observation
                reward = result.reward
                done = result.done
                error = None
            except Exception as e:
                error = str(e)
                reward = 0.0
                done = True
                print(f"[DEBUG] Step error: {error}", flush=True)
            
            # Track metrics for grading
            episode_info['total_cost'] = observation.total_cost
            episode_info['comfort_violations'] = observation.comfort_violations
            episode_info['total_solar_used'] = observation.total_solar_used
            episode_info['steps_completed'] = step
            
            # Track temperature deviation from metadata
            if observation.metadata and 'indoor_temp' in observation.metadata:
                temp_deviations.append(abs(observation.metadata['indoor_temp'] - 22.0))
            
            # Track battery cycles
            battery_action = (action_id % 9) // 3
            if battery_action != last_battery_action and battery_action != 0:
                episode_info['battery_cycles'] = episode_info.get('battery_cycles', 0) + 1
            last_battery_action = battery_action
            
            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            
            # Log step
            log_step(step=step, action=str(action_id), reward=reward, done=done, error=error)
            
            if done:
                break
        
        # Calculate average temperature deviation
        if temp_deviations:
            episode_info['avg_temp_deviation'] = sum(temp_deviations) / len(temp_deviations)
        
        # Grade episode performance
        if grader:
            score = grader(episode_info)
            score = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
            success = score >= task_config.success_threshold
        else:
            # Fallback: normalize total reward
            total_reward = sum(rewards)
            # Estimate max possible reward (rough approximation)
            max_possible = MAX_STEPS * 5.0  # Rough estimate
            score = min(max(total_reward / max_possible, 0.0), 1.0)
            success = score >= 0.5
    
    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        
        # Log end
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())

