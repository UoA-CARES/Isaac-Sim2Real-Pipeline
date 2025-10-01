#!/usr/bin/env python3
"""
Example usage of the Isaac Sim2Real Pipeline for different RL tasks.

This script demonstrates how to use the pipeline with various configurations
and provides templates for common use cases.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import setup_logging, ConfigManager


def example_cartpole_training():
    """Example: Training a cartpole agent using PPO."""
    print("=" * 60)
    print("EXAMPLE 1: CartPole Training with PPO")
    print("=" * 60)
    
    # This would run the actual pipeline
    command = """
    python main.py \\
        --config configs/default_config.yaml \\
        --task cartpole \\
        --algorithm PPO \\
        --experiment-name cartpole_ppo_example
    """
    
    print("Command to run:")
    print(command)
    print("\nThis example would:")
    print("1. Train a PPO agent on CartPole task in Isaac Lab simulation")
    print("2. Use Eureka framework to optimize reward function")
    print("3. Apply domain randomization for sim2real transfer")
    print("4. Evaluate performance on real hardware (simulated)")

