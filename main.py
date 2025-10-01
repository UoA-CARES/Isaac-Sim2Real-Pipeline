#!/usr/bin/env python3
"""
Main entry point for the Isaac Sim2Real RL Pipeline.

# Current only support Rl_Games

This script provides a unified interface to run the complete 3-step pipeline:
1. Simulation training with Isaac Lab
2. LLM-based performance refinement using Eureka framework
3. Sim-to-real transfer with domain randomization and DrEureka

Usage:
    
    # Config
    python main.py * --help  # Show all options
    python main.py * --version  # Show version info
    
    # Stage 1
    python main.py * --list-template-tasks  # List available official template RL tasks from IsaacLab
                   * --create   # Create a new custom RL task environment, your will access a interactive shell to define the task
                   --taskconfig TASKCONFIG  # /home/lee/code/Isaac-Sim2Real-Pipeline/configs/taskconfig.yaml.
                   --simtrain
                   --simtest  
                   --simplay

    # Stage 2
    python main.py --refine  # Run LLM-based reward function refinement using Eureka framework
                   --refineconfig REFINECONFIG  # Path to Eureka configuration file

    # Stage 3


    * not shown on the prototype
"""

import yaml
import argparse
import subprocess
from src.refinement.llm_agent import EurekaAgent


def load_config(config_path):
    """Safely loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Successfully loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: The file was not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return None


def main():
    """
    Main function to load configurations and run the main logic.
    """
    # Set up argument Parser to accept file paths from the command line
    parser = argparse.ArgumentParser(description="TODO:")
    parser.add_argument('--taskconfig', type=str, default="configs/taskconfig.yaml", required=False, help="Path to the task configuration YAML file.")
    parser.add_argument('--simtrain', action='store_true', help="Run simulation training using the provided task configuration")
    parser.add_argument('--simtest', action='store_true', help="Run simulation testing using the provided task configuration")
    parser.add_argument('--simplay', action='store_true', help="Run simulation play using the provided task configuration")
    parser.add_argument('--refine', action='store_true', help="Run LLM-based reward function refinement using Eureka framework")
    parser.add_argument('--refineconfig', type=str, default="configs/refineconfig.yaml", required=False, help="Path to the refine configuration YAML file.")
    
    
    args = parser.parse_args()

    # detect the existence of the task config, if not exist then error:
    if not args.taskconfig:
        print("Error: --taskconfig argument is required.")
        return

    # get workspace, task, num_envs, seed, checkpoint, max_iterations from args or set default values
    task_config = load_config(args.taskconfig)
    
        
    workspace = task_config.get('workspace')
    
    task = task_config.get('task')

    def none_if_str_none(val):
        return None if val == "None" or val is None else int(val)
    num_envs = none_if_str_none(task_config.get('num_envs'))
    seed = none_if_str_none(task_config.get('seed'))
    max_iterations = none_if_str_none(task_config.get('max_iterations'))
    checkpoint = task_config.get('checkpoint') if task_config.get('checkpoint') != "None" else None


    # if the workspace is the official IsaacLab, then cmd_head=["./isaaclab.sh", "-p", "scripts/reinforcement_learning/rl_games/train.py"]
    # if workspace and workspace.rstrip('/').split('/')[-1] == "IsaacLab":
    #     cmd_head = ["./isaaclab.sh", "-p", "scripts/reinforcement_learning/rl_games/train.py"]
    # else:
    # cmd_head = ["python scripts/reinforcement_learning/rl_games/train.py"]

    # simtrain, if simtrain is true, first access the workspace and run 
    # simtrain
    def train_command():
        cmd_head = ["python", "scripts/rl_games/train.py"]
        cmd = cmd_head + ["--task", task]
        
        if num_envs is not None:
            cmd.extend(["--num_envs", str(num_envs)])
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if max_iterations is not None:
            cmd.extend(["--max_iterations", str(max_iterations)])
        if checkpoint is not None:
            cmd.extend(["--checkpoint", checkpoint])
        
        cmd.append("--headless")
        subprocess.run(cmd, cwd=workspace)

    # play command
    def play_command():
        cmd_head_play = ["python", "scripts/rl_games/play.py"]
        cmd = cmd_head_play + ["--task", task]
        if num_envs is not None:
            cmd.extend(["--num_envs", str(num_envs)])
        if checkpoint is not None:
            cmd.extend(["--checkpoint", checkpoint])
        
        subprocess.run(cmd, cwd=workspace)


    if args.simtrain:
        train_command()

    if args.simplay:
        play_command()

    if args.refine:
        from src.refinement.files_operation import load_env_cfg, load_prompts # , write_reward_to_file
        # Eureka refinement
        if not args.refineconfig:
            print("Error: --refineconfig argument is required when --refine is set.")
            return
        refine_config = load_config(args.refineconfig)
        if not refine_config:
            print("Error: Failed to load refine configuration.")
            return
            
        # Read refine config parameters
        iteration = int(refine_config.get('iteration'))
        sample = int(refine_config.get('sample'))
        num_eval = int(refine_config.get('num_eval'))
        env_cfg_path = task_config.get('env_cfg_path')

        
        print(f"Starting Eureka refinement with:")
        print(f"  Iterations: {iteration}")
        print(f"  Samples per iteration: {sample}")
        print(f"  Evaluation episodes: {num_eval}")
        
        # Eureka refinement
        # Init prompts
        env_cfg_dict = load_env_cfg(env_cfg_path)
        prompts_dict = load_prompts()
        prompts_dict["task_description"] = task_config.get('description')
        # Init LLM agents
        llm_agent = EurekaAgent(prompts_dict, env_cfg_dict, agent_config=refine_config.get('agent', {}))
        # Run refinement loop
        for i in range(iteration):
            print(f"Refinement iteration {i+1}/{iteration}")
            for respond_id in range(sample):
                # Sample new reward functions
                reward_func = llm_agent.func_gen()
                # Evaluate the new reward function
                write_reward_to_file(reward_func, file_path, component_path)
                train_command()
            # Select the best reward function
            # Refine the prompt for the next iteration
            prompts = None 



if __name__ == "__main__":
    main()