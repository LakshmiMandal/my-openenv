# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task definitions and graders for Smart Home Energy Management.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    """Configuration for a specific task."""
    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    difficulty: str = Field(..., description="Task difficulty: easy, medium, or hard")
    objective: str = Field(..., description="Clear task objective")
    max_steps: int = Field(96, description="Maximum steps for the task")
    success_threshold: float = Field(..., ge=0.0, le=1.0, description="Score threshold for success")
    
    # Task-specific constraints
    target_cost: Optional[float] = Field(None, description="Target daily cost ($)")
    min_comfort_score: Optional[float] = Field(None, description="Minimum comfort score")
    min_solar_usage: Optional[float] = Field(None, description="Minimum solar usage (kWh)")


# Task 1: Cost Minimization (Easy)
TASK_COST_MINIMIZATION = TaskConfig(
    task_id="cost_minimization",
    name="Cost Minimization",
    difficulty="easy",
    objective="Minimize daily electricity cost while maintaining basic comfort (18-26°C when home)",
    max_steps=96,
    success_threshold=0.6,
    target_cost=5.0,
    min_comfort_score=0.5
)


# Task 2: Comfort Optimization (Medium)
TASK_COMFORT_OPTIMIZATION = TaskConfig(
    task_id="comfort_optimization",
    name="Comfort Optimization",
    difficulty="medium",
    objective="Maintain optimal comfort (20-24°C) while keeping costs under $8/day",
    max_steps=96,
    success_threshold=0.7,
    target_cost=8.0,
    min_comfort_score=0.8
)


# Task 3: Sustainability Maximization (Hard)
TASK_SUSTAINABILITY = TaskConfig(
    task_id="sustainability",
    name="Sustainability Maximization",
    difficulty="hard",
    objective="Maximize solar energy usage (>15 kWh/day), maintain comfort (20-24°C), and minimize cost (<$6/day)",
    max_steps=96,
    success_threshold=0.8,
    target_cost=6.0,
    min_comfort_score=0.75,
    min_solar_usage=15.0
)


# All tasks registry
ALL_TASKS = [
    TASK_COST_MINIMIZATION,
    TASK_COMFORT_OPTIMIZATION,
    TASK_SUSTAINABILITY
]


def grade_cost_minimization(episode_info: Dict[str, Any]) -> float:
    """
    Grade Task 1: Cost Minimization
    
    Scoring:
    - 50% based on cost (lower is better, target <$5)
    - 30% based on comfort violations (fewer is better)
    - 20% based on completing the episode
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    total_cost = episode_info.get('total_cost', 10.0)
    comfort_violations = episode_info.get('comfort_violations', 96)
    steps_completed = episode_info.get('steps_completed', 0)
    
    # Cost score (0-0.5)
    cost_score = max(0.0, min(0.5, 0.5 * (1.0 - (total_cost - 5.0) / 5.0)))
    
    # Comfort score (0-0.3)
    comfort_score = max(0.0, 0.3 * (1.0 - comfort_violations / 96.0))
    
    # Completion score (0-0.2)
    completion_score = 0.2 * (steps_completed / 96.0)
    
    total_score = cost_score + comfort_score + completion_score
    return min(1.0, max(0.0, total_score))


def grade_comfort_optimization(episode_info: Dict[str, Any]) -> float:
    """
    Grade Task 2: Comfort Optimization
    
    Scoring:
    - 60% based on comfort (fewer violations, tighter temperature control)
    - 30% based on cost (must be under $8)
    - 10% based on completing the episode
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    total_cost = episode_info.get('total_cost', 15.0)
    comfort_violations = episode_info.get('comfort_violations', 96)
    steps_completed = episode_info.get('steps_completed', 0)
    avg_temp_deviation = episode_info.get('avg_temp_deviation', 5.0)
    
    # Comfort score (0-0.6)
    violation_score = 0.3 * (1.0 - comfort_violations / 96.0)
    temp_score = 0.3 * max(0.0, 1.0 - avg_temp_deviation / 4.0)
    comfort_score = violation_score + temp_score
    
    # Cost score (0-0.3)
    if total_cost <= 8.0:
        cost_score = 0.3
    else:
        cost_score = max(0.0, 0.3 * (1.0 - (total_cost - 8.0) / 4.0))
    
    # Completion score (0-0.1)
    completion_score = 0.1 * (steps_completed / 96.0)
    
    total_score = comfort_score + cost_score + completion_score
    return min(1.0, max(0.0, total_score))


def grade_sustainability(episode_info: Dict[str, Any]) -> float:
    """
    Grade Task 3: Sustainability Maximization
    
    Scoring:
    - 40% based on solar energy usage (target >15 kWh)
    - 30% based on cost (target <$6)
    - 20% based on comfort (target 20-24°C)
    - 10% based on battery efficiency
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    total_cost = episode_info.get('total_cost', 10.0)
    comfort_violations = episode_info.get('comfort_violations', 96)
    total_solar_used = episode_info.get('total_solar_used', 0.0)
    battery_cycles = episode_info.get('battery_cycles', 10)
    steps_completed = episode_info.get('steps_completed', 0)
    
    # Solar score (0-0.4)
    if total_solar_used >= 15.0:
        solar_score = 0.4
    else:
        solar_score = max(0.0, 0.4 * (total_solar_used / 15.0))
    
    # Cost score (0-0.3)
    if total_cost <= 6.0:
        cost_score = 0.3
    else:
        cost_score = max(0.0, 0.3 * (1.0 - (total_cost - 6.0) / 4.0))
    
    # Comfort score (0-0.2)
    comfort_score = 0.2 * (1.0 - comfort_violations / 96.0)
    
    # Battery efficiency score (0-0.1)
    battery_score = max(0.0, 0.1 * (1.0 - battery_cycles / 20.0))
    
    total_score = solar_score + cost_score + comfort_score + battery_score
    
    # Bonus for completing the full episode
    if steps_completed >= 96:
        total_score *= 1.05
    
    return min(1.0, max(0.0, total_score))


def get_grader(task_id: str):
    """Get the grader function for a specific task."""
    graders = {
        'cost_minimization': grade_cost_minimization,
        'comfort_optimization': grade_comfort_optimization,
        'sustainability': grade_sustainability
    }
    return graders.get(task_id)


def get_task_config(task_id: str) -> TaskConfig:
    """Get the configuration for a specific task."""
    for task in ALL_TASKS:
        if task.task_id == task_id:
            return task
    raise ValueError(f"Unknown task_id: {task_id}")


