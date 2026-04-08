# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Smart Home Energy Management Environment.

The smart home environment simulates energy management with HVAC, battery, and appliances.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional


class SmartHomeAction(Action):
    """Action for the Smart Home Energy Management environment."""

    action_id: int = Field(..., ge=0, le=26, description="Discrete action ID (0-26)")
    hvac: Optional[int] = Field(None, ge=0, le=2, description="HVAC control (0=off, 1=heat, 2=cool)")
    battery: Optional[int] = Field(None, ge=0, le=2, description="Battery control (0=idle, 1=charge, 2=discharge)")
    appliances: Optional[int] = Field(None, ge=0, le=2, description="Appliance usage (0=minimal, 1=normal, 2=high)")


class SmartHomeObservation(Observation):
    """Observation from the Smart Home Energy Management environment."""

    hour_of_day: float = Field(..., ge=0.0, le=1.0, description="Normalized hour (0-23)")
    day_of_week: float = Field(..., ge=0.0, le=1.0, description="Normalized day (0-6)")
    outdoor_temp: float = Field(..., ge=0.0, le=1.0, description="Normalized outdoor temperature")
    indoor_temp: float = Field(..., ge=0.0, le=1.0, description="Normalized indoor temperature")
    solar_generation: float = Field(..., ge=0.0, le=1.0, description="Normalized solar generation")
    battery_charge: float = Field(..., ge=0.0, le=1.0, description="Normalized battery charge")
    electricity_price: float = Field(..., ge=0.0, le=1.0, description="Normalized electricity price")
    occupancy: float = Field(..., ge=0.0, le=1.0, description="Home occupancy (0 or 1)")
    hvac_status: float = Field(..., ge=0.0, le=1.0, description="Normalized HVAC status")
    
    # Additional info fields
    total_cost: float = Field(default=0.0, description="Total cost so far")
    total_solar_used: float = Field(default=0.0, description="Total solar energy used")
    comfort_violations: int = Field(default=0, description="Number of comfort violations")


