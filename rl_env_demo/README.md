---
title: Smart Home Energy Management Environment
emoji: 🏠
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - energy-management
  - smart-home
  - reinforcement-learning
  - sustainability
---

# Smart Home Energy Management Environment

A complete, real-world OpenEnv environment for training AI agents to optimize energy consumption in smart homes.

## Overview

The **Smart Home Energy Management Environment** simulates a realistic smart home where an AI agent learns to:
- Optimize energy costs by managing HVAC, appliances, and battery storage
- Balance comfort (temperature control) with cost efficiency
- Maximize solar energy utilization
- Make decisions based on time-of-day pricing, weather, and occupancy patterns

## Features

- ✅ **Standard OpenEnv API**: Compatible with OpenEnv client/server architecture
- ✅ **Real-world dynamics**: Solar generation, battery storage, dynamic pricing
- ✅ **Multi-objective optimization**: Cost, comfort, and sustainability
- ✅ **Comprehensive observation space**: 9 normalized features
- ✅ **Discrete action space**: 27 possible actions (HVAC × Battery × Appliances)
- ✅ **Episode-based learning**: 24-hour episodes (96 steps of 15 minutes each)
- ✅ **Detailed metrics**: Cost tracking, solar usage, comfort violations
- ✅ **Three difficulty levels**: Easy, Medium, and Hard tasks with programmatic graders

## Quick Start

The simplest way to use the Smart Home Energy Management environment is through the `SmartHomeEnv` class:

```python
from rl_env_demo import SmartHomeAction, SmartHomeEnv

try:
    # Create environment from Docker image
    env = SmartHomeEnv.from_docker_image("rl_env_demo-env:latest")

    # Reset
    result = env.reset()
    print(f"Initial indoor temp: {result.observation.indoor_temp * 15 + 15:.1f}°C")

    # Run episode
    for step in range(96):  # 24 hours
        # Simple strategy: minimize cost
        action = SmartHomeAction(action_id=0)  # All systems off
        result = env.step(action)
        
        if step % 16 == 0:  # Print every 4 hours
            print(f"Step {step}: Reward={result.reward:.2f}, Cost=${result.observation.total_cost:.2f}")
        
        if result.done:
            break

finally:
    env.close()
```

## Building the Docker Image

Before using the environment, build the Docker image:

```bash
# From project root
docker build -t rl_env_demo-env:latest -f Dockerfile .
```

## Environment Details

### Observation Space

A 9-dimensional continuous space (all values normalized to [0, 1]):

| Feature | Range | Description |
|---------|-------|-------------|
| hour_of_day | 0-23 | Current hour |
| day_of_week | 0-6 | Current day (0=Monday) |
| outdoor_temp | -20 to 45°C | Outside temperature |
| indoor_temp | 15 to 30°C | Inside temperature |
| solar_generation | 0-10 kW | Solar panel output |
| battery_charge | 0-100% | Battery state of charge |
| electricity_price | $0.05-0.50/kWh | Current electricity price |
| occupancy | 0 or 1 | Home occupied (1) or empty (0) |
| hvac_status | 0, 1, or 2 | HVAC state (off/heating/cooling) |

### Action Space

Discrete space with 27 actions, encoded as combinations of:

- **HVAC Control**: 0=off, 1=heating, 2=cooling
- **Battery Management**: 0=idle, 1=charge, 2=discharge  
- **Appliance Usage**: 0=minimal, 1=normal, 2=high

Action encoding: `action_id = hvac * 9 + battery * 3 + appliances`

Examples:
- Action 0: All systems off (HVAC=0, Battery=0, Appliances=0)
- Action 13: All systems mid (HVAC=1, Battery=1, Appliances=1)
- Action 26: All systems max (HVAC=2, Battery=2, Appliances=2)

### Reward Function

The reward balances multiple objectives:

1. **Cost Minimization**: Negative reward for electricity purchased from grid
2. **Comfort Maintenance**: Positive reward for keeping temperature in 20-24°C range when home
3. **Solar Utilization**: Bonus for using solar energy
4. **Battery Health**: Penalties for overcharging/over-discharging
5. **Smart Battery Usage**: Bonuses for charging during solar surplus and discharging during peak prices

## Tasks and Objectives

This environment includes three tasks with increasing difficulty:

### Task 1: Cost Minimization (Easy)
**Objective:** Minimize daily electricity cost while maintaining basic comfort (18-26°C when home)

- **Difficulty:** Easy
- **Success Threshold:** 0.6
- **Target Metrics:**
  - Daily cost under $5
  - Comfort violations < 20% of occupied time
  - Complete 24-hour episode

### Task 2: Comfort Optimization (Medium)
**Objective:** Maintain optimal comfort (20-24°C) while keeping costs under $8/day

- **Difficulty:** Medium
- **Success Threshold:** 0.7
- **Target Metrics:**
  - Maintain 20-24°C when home (comfort score > 0.8)
  - Daily cost under $8
  - Minimize temperature deviation from 22°C target

### Task 3: Sustainability Maximization (Hard)
**Objective:** Maximize solar energy usage (>15 kWh/day), maintain comfort (20-24°C), and minimize cost (<$6/day)

- **Difficulty:** Hard
- **Success Threshold:** 0.8
- **Target Metrics:**
  - Solar energy usage > 15 kWh/day
  - Daily cost under $6
  - Comfort score > 0.75
  - Efficient battery usage (minimize cycles)

## Advanced Usage

### Rule-Based Agent Example

```python
from rl_env_demo import SmartHomeAction, SmartHomeEnv

with SmartHomeEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    
    for step in range(96):
        obs = result.observation
        
        # Denormalize key features
        indoor_temp = obs.indoor_temp * 15.0 + 15.0
        battery_charge = obs.battery_charge * 100.0
        electricity_price = obs.electricity_price * 0.45 + 0.05
        occupancy = int(obs.occupancy)
        
        # Rule-based logic
        if occupancy and indoor_temp < 20:
            hvac = 1  # Heat when cold and home
        elif occupancy and indoor_temp > 24:
            hvac = 2  # Cool when hot and home
        else:
            hvac = 0
        
        if electricity_price > 0.25 and battery_charge > 20:
            battery = 2  # Discharge during peak pricing
        elif obs.solar_generation > 0.3 and battery_charge < 90:
            battery = 1  # Charge from solar
        else:
            battery = 0
        
        appliances = 1 if occupancy else 0
        
        action_id = hvac * 9 + battery * 3 + appliances
        result = env.step(SmartHomeAction(action_id=action_id))
        
        if result.done:
            print(f"Episode complete! Total cost: ${obs.total_cost:.2f}")
            break
```

## Deploying to Hugging Face Spaces

Deploy your environment to Hugging Face Spaces:

```bash
# From the environment directory
openenv push

# Or specify options
openenv push --namespace my-org --private
```

After deployment, your space will be available with:
- **Web Interface** at `/web` - Interactive UI
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **WebSocket** at `/ws` - Persistent session endpoint

## Environment Dynamics

### Solar Generation
- Peaks at noon with realistic daily patterns
- Weather variability (clouds reduce output)
- Zero generation at night

### Electricity Pricing
- **Peak hours** (6-9 AM, 5-9 PM): $0.35/kWh
- **Off-peak** (9 PM-6 AM): $0.10/kWh
- **Mid-peak** (other times): $0.20/kWh

### Temperature Dynamics
- HVAC heating/cooling rate: ±2°C per hour
- Natural drift toward outdoor temperature
- Outdoor temperature varies by time of day

### Battery
- Capacity: 13.5 kWh (Tesla Powerwall size)
- Max charge/discharge rate: 5 kW
- Efficiency: 90% (implicit in reward function)

### Occupancy Patterns
- Weekdays: Away 8 AM-5 PM (80% probability)
- Weekends: Home most of the time (90% probability)

## Development & Testing

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

### Direct Environment Testing

Test the environment logic directly:

```bash
python3 server/smart_home_environment.py
```

## Project Structure

```
rl_env_demo/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies
├── client.py              # SmartHomeEnv client
├── models.py              # Action and Observation models
├── tasks.py               # Task definitions and graders
└── server/
    ├── __init__.py        # Server module exports
    ├── smart_home_environment.py  # Core environment logic
    ├── app.py             # FastAPI application
    └── Dockerfile         # Container image definition
```

## License

BSD-style license - See LICENSE file for details.

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{smart_home_energy_env,
  title={Smart Home Energy Management Environment},
  author={OpenEnv Team},
  year={2026},
  url={https://github.com/yourusername/rl_env_demo}
}
