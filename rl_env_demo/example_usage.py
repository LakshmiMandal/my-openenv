"""
Example usage of the Smart Home Energy Management Environment.

This script demonstrates how to interact with the environment using different strategies.
"""

import numpy as np
from rl_env_demo import SmartHomeAction, SmartHomeEnv


def random_agent_example():
    """Example: Random agent taking random actions."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Random Agent")
    print("="*70)
    
    # Connect to environment (assumes server is running)
    with SmartHomeEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        print(f"\nInitial state:")
        print(f"  Indoor temp: {result.observation.indoor_temp * 15 + 15:.1f}°C")
        print(f"  Battery: {result.observation.battery_charge * 100:.1f}%")
        
        total_reward = 0
        for step in range(10):
            # Random action
            action_id = np.random.randint(0, 27)
            action = SmartHomeAction(action_id=action_id)
            
            result = env.step(action)
            total_reward += result.reward
            
            print(f"\nStep {step + 1}:")
            print(f"  Action: {action_id}")
            print(f"  Reward: {result.reward:.2f}")
            print(f"  Total Cost: ${result.observation.total_cost:.2f}")
            
            if result.done:
                break
        
        print(f"\nTotal reward: {total_reward:.2f}")


def rule_based_agent_example():
    """Example: Rule-based agent with simple heuristics."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Rule-Based Agent")
    print("="*70)
    print("Rules:")
    print("  - Use HVAC to maintain 20-24°C when home")
    print("  - Charge battery when solar > consumption")
    print("  - Discharge battery during peak pricing")
    print("  - Minimize appliance usage when away")
    
    with SmartHomeEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        
        total_reward = 0
        for step in range(96):  # Full 24-hour episode
            obs = result.observation
            
            # Denormalize key features
            indoor_temp = obs.indoor_temp * 15.0 + 15.0
            battery_charge = obs.battery_charge * 100.0
            electricity_price = obs.electricity_price * 0.45 + 0.05
            solar_gen = obs.solar_generation * 10.0
            occupancy = int(obs.occupancy)
            
            # Rule-based logic
            # HVAC control
            if occupancy:
                if indoor_temp < 20.0:
                    hvac = 1  # Heating
                elif indoor_temp > 24.0:
                    hvac = 2  # Cooling
                else:
                    hvac = 0  # Off
            else:
                hvac = 0  # Off when away
            
            # Battery management
            if solar_gen > 2.0 and battery_charge < 90:
                battery = 1  # Charge from solar
            elif electricity_price > 0.25 and battery_charge > 20:
                battery = 2  # Discharge during peak pricing
            else:
                battery = 0  # Idle
            
            # Appliance usage
            if occupancy:
                if electricity_price < 0.15:
                    appliances = 2  # High usage during cheap electricity
                else:
                    appliances = 1  # Normal usage
            else:
                appliances = 0  # Minimal when away
            
            # Encode action
            action_id = hvac * 9 + battery * 3 + appliances
            action = SmartHomeAction(action_id=action_id)
            
            result = env.step(action)
            total_reward += result.reward
            
            # Print every 4 hours
            if step % 16 == 0:
                print(f"\nStep {step + 1}:")
                print(f"  Occupancy: {'Home' if occupancy else 'Away'}")
                print(f"  Indoor: {indoor_temp:.1f}°C, HVAC: {['Off', 'Heat', 'Cool'][hvac]}")
                print(f"  Solar: {solar_gen:.2f}kW, Battery: {battery_charge:.1f}%")
                print(f"  Price: ${electricity_price:.2f}/kWh")
                print(f"  Reward: {result.reward:.2f}, Total: {total_reward:.2f}")
            
            if result.done:
                break
        
        print(f"\n{'='*70}")
        print(f"Episode Summary:")
        print(f"  Total Steps: {step + 1}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Total Cost: ${result.observation.total_cost:.2f}")
        print(f"  Solar Used: {result.observation.total_solar_used:.2f} kWh")
        print(f"  Comfort Violations: {result.observation.comfort_violations}")
        print(f"{'='*70}")


def cost_minimization_strategy():
    """Example: Strategy focused on minimizing cost."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Cost Minimization Strategy")
    print("="*70)
    
    with SmartHomeEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        
        total_reward = 0
        for step in range(96):
            obs = result.observation
            
            # Denormalize
            indoor_temp = obs.indoor_temp * 15.0 + 15.0
            battery_charge = obs.battery_charge * 100.0
            electricity_price = obs.electricity_price * 0.45 + 0.05
            occupancy = int(obs.occupancy)
            
            # Minimal HVAC usage
            if occupancy and indoor_temp < 18:
                hvac = 1
            elif occupancy and indoor_temp > 26:
                hvac = 2
            else:
                hvac = 0
            
            # Smart battery usage
            if electricity_price > 0.25 and battery_charge > 20:
                battery = 2  # Discharge during expensive hours
            elif obs.solar_generation > 0.3 and battery_charge < 90:
                battery = 1  # Charge from solar
            else:
                battery = 0
            
            # Minimal appliances
            appliances = 0 if not occupancy else 1
            
            action_id = hvac * 9 + battery * 3 + appliances
            result = env.step(SmartHomeAction(action_id=action_id))
            total_reward += result.reward
            
            if result.done:
                break
        
        print(f"\nFinal Results:")
        print(f"  Total Cost: ${result.observation.total_cost:.2f}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Comfort Violations: {result.observation.comfort_violations}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Smart Home Energy Management Environment - Examples")
    print("="*70)
    print("\nMake sure the server is running:")
    print("  uvicorn server.app:app --reload")
    print("\nOr use Docker:")
    print("  docker run -p 8000:8000 rl_env_demo-env:latest")
    
    try:
        random_agent_example()
        rule_based_agent_example()
        cost_minimization_strategy()
        
        print("\n" + "="*70)
        print("All examples completed!")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the server is running on http://localhost:8000")


if __name__ == "__main__":
    main()


