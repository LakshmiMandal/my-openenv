# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smart Home Energy Management Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SmartHomeAction, SmartHomeObservation
except ImportError:
    from models import SmartHomeAction, SmartHomeObservation


class SmartHomeEnv(
    EnvClient[SmartHomeAction, SmartHomeObservation, State]
):
    """
    Client for the Smart Home Energy Management Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SmartHomeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(f"Indoor temp: {result.observation.indoor_temp}")
        ...
        ...     # Control HVAC, battery, and appliances
        ...     action = SmartHomeAction(action_id=13)  # Mid settings
        ...     result = client.step(action)
        ...     print(f"Reward: {result.reward}")

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SmartHomeEnv.from_docker_image("rl_env_demo-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     for _ in range(96):  # 24 hours
        ...         action = SmartHomeAction(action_id=0)  # All off
        ...         result = client.step(action)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SmartHomeAction) -> Dict:
        """
        Convert SmartHomeAction to JSON payload for step message.

        Args:
            action: SmartHomeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_id": action.action_id,
            "hvac": action.hvac,
            "battery": action.battery,
            "appliances": action.appliances,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SmartHomeObservation]:
        """
        Parse server response into StepResult[SmartHomeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SmartHomeObservation
        """
        obs_data = payload.get("observation", {})
        observation = SmartHomeObservation(
            hour_of_day=obs_data.get("hour_of_day", 0.0),
            day_of_week=obs_data.get("day_of_week", 0.0),
            outdoor_temp=obs_data.get("outdoor_temp", 0.0),
            indoor_temp=obs_data.get("indoor_temp", 0.0),
            solar_generation=obs_data.get("solar_generation", 0.0),
            battery_charge=obs_data.get("battery_charge", 0.0),
            electricity_price=obs_data.get("electricity_price", 0.0),
            occupancy=obs_data.get("occupancy", 0.0),
            hvac_status=obs_data.get("hvac_status", 0.0),
            total_cost=obs_data.get("total_cost", 0.0),
            total_solar_used=obs_data.get("total_solar_used", 0.0),
            comfort_violations=obs_data.get("comfort_violations", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


