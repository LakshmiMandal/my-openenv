# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smart Home Energy Management Environment Implementation.

A real-world environment where an AI agent learns to optimize energy consumption
in a smart home by controlling appliances, HVAC, and battery storage.
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional
from uuid import uuid4
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import SmartHomeAction, SmartHomeObservation
except ImportError:
    from rl_env_demo.models import SmartHomeAction, SmartHomeObservation


class RlEnvDemoEnvironment(Environment):
    """
    Smart Home Energy Management Environment
    
    Observation Space:
        - hour_of_day: 0-23 (normalized to 0-1)
        - day_of_week: 0-6 (normalized to 0-1)
        - outdoor_temp: -20 to 45°C (normalized)
        - indoor_temp: 15 to 30°C (normalized)
        - solar_generation: 0-10 kW (normalized)
        - battery_charge: 0-100% (normalized)
        - electricity_price: 0.05-0.50 $/kWh (normalized)
        - occupancy: 0 or 1 (home/away)
        - hvac_status: 0 (off), 1 (heating), 2 (cooling) (normalized)
        
    Action Space (Discrete with 27 combinations):
        - HVAC: 0=off, 1=heating, 2=cooling
        - Battery: 0=idle, 1=charge, 2=discharge
        - Appliances: 0=minimal, 1=normal, 2=high_usage
    """
    
    # Constants
    MAX_STEPS = 96  # 24 hours in 15-min intervals
    BATTERY_CAPACITY = 13.5  # kWh (Tesla Powerwall size)
    SOLAR_MAX = 10.0  # kW peak generation
    
    # Appliance power consumption (kW)
    APPLIANCE_POWER = {
        'minimal': 0.5,
        'normal': 2.0,
        'high_usage': 4.0
    }
    
    # HVAC power consumption (kW)
    HVAC_POWER = {
        'off': 0.0,
        'heating': 3.5,
        'cooling': 3.0
    }
    
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the smart home environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Environment state
        self.hour = 0
        self.day = 0
        self.outdoor_temp = 20.0
        self.indoor_temp = 22.0
        self.solar_generation = 0.0
        self.battery_charge = 50.0
        self.electricity_price = 0.15
        self.occupancy = 1
        self.hvac_status = 0
        
        # Tracking
        self.total_cost = 0.0
        self.total_solar_used = 0.0
        self.comfort_violations = 0
    
    def _decode_action(self, action_id: int) -> Tuple[int, int, int]:
        """Decode discrete action into HVAC, Battery, and Appliance settings."""
        hvac = action_id // 9
        battery = (action_id % 9) // 3
        appliances = action_id % 3
        return hvac, battery, appliances
    
    def _get_electricity_price(self, hour: int) -> float:
        """Get electricity price based on time of day."""
        if 6 <= hour < 9 or 17 <= hour < 21:  # Peak hours
            return 0.35
        elif 21 <= hour < 24 or 0 <= hour < 6:  # Off-peak hours
            return 0.10
        else:  # Mid-peak hours
            return 0.20
    
    def _get_solar_generation(self, hour: int, day: int) -> float:
        """Simulate solar panel generation."""
        if 6 <= hour <= 18:  # Daylight hours
            hour_factor = 1.0 - abs(hour - 12) / 6.0
            weather_factor = np.random.uniform(0.6, 1.0)
            return self.SOLAR_MAX * hour_factor * weather_factor
        return 0.0
    
    def _get_outdoor_temp(self, hour: int, day: int) -> float:
        """Simulate outdoor temperature variation."""
        base_temp = 20.0 + np.random.uniform(-5, 5)
        hour_variation = 5.0 * np.sin((hour - 6) * np.pi / 12)
        return base_temp + hour_variation
    
    def _get_occupancy(self, hour: int, day: int) -> int:
        """Simulate home occupancy patterns."""
        if day < 5:  # Monday-Friday
            if 8 <= hour < 17:  # Work hours
                return 0 if np.random.random() < 0.8 else 1
            else:
                return 1
        else:  # Weekend
            return 1 if np.random.random() < 0.9 else 0
    
    def _update_indoor_temp(self, hvac_action: int, time_step: float = 0.25) -> None:
        """Update indoor temperature based on HVAC action."""
        if hvac_action == 1:  # Heating
            self.indoor_temp += 2.0 * time_step
        elif hvac_action == 2:  # Cooling
            self.indoor_temp -= 2.0 * time_step
        
        # Natural temperature drift
        temp_diff = self.outdoor_temp - self.indoor_temp
        self.indoor_temp += 0.5 * temp_diff * time_step
        
        # Clamp to reasonable range
        self.indoor_temp = np.clip(self.indoor_temp, 15.0, 30.0)
    
    def _calculate_power_consumption(self, hvac: int, battery: int, appliances: int) -> float:
        """Calculate total power consumption in kW."""
        hvac_power = self.HVAC_POWER[['off', 'heating', 'cooling'][hvac]]
        appliance_power = self.APPLIANCE_POWER[['minimal', 'normal', 'high_usage'][appliances]]
        return hvac_power + appliance_power
    
    def _calculate_reward(self, hvac: int, battery: int, appliances: int,
                         power_consumed: float, grid_power: float) -> float:
        """Calculate reward balancing cost, comfort, and sustainability."""
        reward = 0.0
        
        # 1. Cost penalty
        cost = grid_power * self.electricity_price * 0.25
        reward -= cost * 10
        self.total_cost += cost
        
        # 2. Comfort reward/penalty
        if self.occupancy == 1:
            target_temp = 22.0
            temp_diff = abs(self.indoor_temp - target_temp)
            if temp_diff < 1.0:
                reward += 2.0
            elif temp_diff < 2.0:
                reward += 0.5
            else:
                reward -= temp_diff * 2.0
                self.comfort_violations += 1
        
        # 3. Solar usage bonus
        solar_used = min(power_consumed, self.solar_generation)
        reward += solar_used * 0.5
        self.total_solar_used += solar_used * 0.25
        
        # 4. Battery health penalty
        if battery == 1 and self.battery_charge > 90:
            reward -= 0.5
        elif battery == 2 and self.battery_charge < 10:
            reward -= 0.5
        
        # 5. Smart battery usage bonus
        if battery == 1 and self.solar_generation > power_consumed:
            reward += 1.0
        elif battery == 2 and self.electricity_price > 0.25:
            reward += 1.0
        
        return reward
    
    def _get_observation(self) -> SmartHomeObservation:
        """Get normalized observation."""
        return SmartHomeObservation(
            hour_of_day=self.hour / 23.0,
            day_of_week=self.day / 6.0,
            outdoor_temp=(self.outdoor_temp + 20) / 65.0,
            indoor_temp=(self.indoor_temp - 15) / 15.0,
            solar_generation=self.solar_generation / self.SOLAR_MAX,
            battery_charge=self.battery_charge / 100.0,
            electricity_price=(self.electricity_price - 0.05) / 0.45,
            occupancy=float(self.occupancy),
            hvac_status=self.hvac_status / 2.0,
            total_cost=self.total_cost,
            total_solar_used=self.total_solar_used,
            comfort_violations=self.comfort_violations,
            done=False,
            reward=0.0
        )
    
    def reset(self) -> SmartHomeObservation:
        """Reset the environment to initial state."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Reset time
        self.hour = np.random.randint(0, 24)
        self.day = np.random.randint(0, 7)
        
        # Reset environment conditions
        self.outdoor_temp = self._get_outdoor_temp(self.hour, self.day)
        self.indoor_temp = 22.0 + np.random.uniform(-2, 2)
        self.solar_generation = self._get_solar_generation(self.hour, self.day)
        self.battery_charge = np.random.uniform(30, 70)
        self.electricity_price = self._get_electricity_price(self.hour)
        self.occupancy = self._get_occupancy(self.hour, self.day)
        self.hvac_status = 0
        
        # Reset tracking
        self.total_cost = 0.0
        self.total_solar_used = 0.0
        self.comfort_violations = 0
        
        observation = self._get_observation()
        observation.done = False
        observation.reward = 0.0
        
        return observation
    
    def step(self, action: SmartHomeAction) -> SmartHomeObservation:
        """Execute one time step in the environment."""
        self._state.step_count += 1
        
        # Decode action
        hvac, battery, appliances = self._decode_action(action.action_id)
        
        # Calculate power consumption
        power_consumed = self._calculate_power_consumption(hvac, battery, appliances)
        
        # Solar generation
        net_power = power_consumed - self.solar_generation
        
        # Battery management
        battery_power = 0.0
        if battery == 1:  # Charge battery
            charge_power = min(5.0, self.solar_generation - power_consumed)
            if charge_power > 0:
                charge_amount = charge_power * 0.25
                available_capacity = (100 - self.battery_charge) / 100 * self.BATTERY_CAPACITY
                actual_charge = min(charge_amount, available_capacity)
                self.battery_charge += (actual_charge / self.BATTERY_CAPACITY) * 100
                battery_power = -actual_charge / 0.25
        elif battery == 2:  # Discharge battery
            discharge_power = min(5.0, net_power)
            if discharge_power > 0 and self.battery_charge > 5:
                discharge_amount = discharge_power * 0.25
                available_energy = self.battery_charge / 100 * self.BATTERY_CAPACITY
                actual_discharge = min(discharge_amount, available_energy)
                self.battery_charge -= (actual_discharge / self.BATTERY_CAPACITY) * 100
                battery_power = actual_discharge / 0.25
        
        # Grid power
        grid_power = max(0, net_power - battery_power)
        
        # Update indoor temperature
        self._update_indoor_temp(hvac)
        self.hvac_status = hvac
        
        # Calculate reward
        reward = self._calculate_reward(hvac, battery, appliances, power_consumed, grid_power)
        
        # Update time
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day = (self.day + 1) % 7
        
        # Update environment conditions
        self.outdoor_temp = self._get_outdoor_temp(self.hour, self.day)
        self.solar_generation = self._get_solar_generation(self.hour, self.day)
        self.electricity_price = self._get_electricity_price(self.hour)
        self.occupancy = self._get_occupancy(self.hour, self.day)
        
        # Get observation
        observation = self._get_observation()
        observation.reward = reward
        observation.done = self._state.step_count >= self.MAX_STEPS
        observation.metadata = {
            'hour': self.hour,
            'day': self.day,
            'indoor_temp': self.indoor_temp,
            'outdoor_temp': self.outdoor_temp,
            'battery_charge': self.battery_charge,
            'solar_generation': self.solar_generation,
            'electricity_price': self.electricity_price,
            'power_consumed': power_consumed,
            'grid_power': grid_power,
            'cost': grid_power * self.electricity_price * 0.25,
            'step': self._state.step_count
        }
        
        return observation
    
    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

