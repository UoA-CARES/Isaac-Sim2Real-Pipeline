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


# TODO: Evaluation
# TODO: More Agentic
# TODO: CKPT and store

import os
import yaml
import argparse
import subprocess
from src.refinement.llm_agent import EurekaAgent
from src.refinement.files_operation import load_env_cfg, load_prompts, write_code_to_file, read_tb, get_latest_checkpoint_dir, summarize_tensorboard
import shutil


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

    # Load task configuration
    task_config = load_config(args.taskconfig)
    workspace = task_config.get('workspace')
    env_cfg_path = os.path.join(workspace, task_config.get('env_cfg_path'))
    logs_path = os.path.join(workspace, task_config.get('logs_path'))
    task_description = task_config.get('description')

    # command parameters
    task = task_config.get('task')
    checkpoint = task_config.get('checkpoint') 
    whichpython = task_config.get('python_env') 
    num_envs = task_config.get('num_envs')
    seed = task_config.get('seed')
    max_iterations = task_config.get('max_iterations')

    # if the workspace is the official IsaacLab, then cmd_head=["./isaaclab.sh", "-p", "scripts/reinforcement_learning/rl_games/train.py"]
    # if workspace and workspace.rstrip('/').split('/')[-1] == "IsaacLab":
    #     cmd_head = ["./isaaclab.sh", "-p", "scripts/reinforcement_learning/rl_games/train.py"]
    # else:
    # cmd_head = ["python scripts/reinforcement_learning/rl_games/train.py"]
 
    # simtrain, if simtrain is true, first access the workspace and run 
    # simtrain
    def train_command(whichpython="python", silence=False, log_name=None):
        try:
            # OUTPUT: log_path (if None then the training is no sccusseful), consecutive successes, best checkpoint path
            cmd_head = [whichpython, "scripts/rl_games/train.py"]
            cmd = cmd_head + ["--task", task]

            if num_envs != "None":
                cmd.extend(["--num_envs", str(num_envs)])
            if seed != "None":
                cmd.extend(["--seed", str(seed)])
            if max_iterations != "None":
                cmd.extend(["--max_iterations", str(max_iterations)])
            if checkpoint != "None":
                cmd.extend(["--checkpoint", checkpoint])
            # Run the command in the specified workspace 
            cmd.append("--headless")
            # TODO: silence
            # if silence:
            #     cmd.extend(["--verbose > /dev/null 2>&1"])  # --verbose > /dev/null 2>&1
            subprocess.run(cmd, cwd=workspace)
            # Get the latest log path
            log_path = get_latest_checkpoint_dir(logs_path=logs_path)

            if log_name is not None:
                # Rename the log_path folder to log_name
                new_log_path = os.path.join(os.path.dirname(log_path), log_name)
                os.rename(log_path, new_log_path)
                log_path = new_log_path

            os.makedirs(os.path.join(log_path, "training_record"), exist_ok=True)
            
            tb_path = os.path.join(log_path, "summaries", os.listdir(os.path.join(log_path, "summaries"))[0])
            # Summarize the tensorboard file for LLM analysis
            summarize_tensorboard(tb_path, os.path.join(log_path, "training_record", "training_summary.txt"))
            # Consecutive Successes
            consecutive_successes_events = read_tb(tb_path, 'Episode/consecutive_successes')
            max_con_successes = max(event.value for event in consecutive_successes_events)
            return log_path, max_con_successes, tb_path

        except Exception as e:
            print(f"Error during training subprocess: {e}")
            return None, -2**31, None
        

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
        log_path, max_con_successes, tb_path = train_command(whichpython=whichpython, silence=False, log_name="testtest")
        print(f"Training completed.")
        print(f"  Log path: {log_path}")
        print(f"  Max Consecutive Successes: {max_con_successes}")
        print(f"  TensorBoard path: {tb_path}")

    if args.simplay:
        play_command()

    if args.refine:
        # TODO: delete all the folder under: /home/lee/code/isaactasks/ant/logs/rl_games/ant_direct
        # Clear previous logs if running refinement
        logs_dir = "/home/lee/code/isaactasks/ant/logs/rl_games/ant_direct"
        if os.path.exists(logs_dir):
            print(f"Clearing previous logs at {logs_dir}")
            for item in os.listdir(logs_dir):
                item_path = os.path.join(logs_dir, item)
                if os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path)
                        print(f"  Deleted: {item}")
                    except Exception as e:
                        print(f"  Failed to delete {item}: {e}")
        # Run git checkout . under /home/lee/code/isaactasks
        print("Running git checkout . under /home/lee/code/isaactasks")
        subprocess.run(["git", "checkout", "."], cwd="/home/lee/code/isaactasks")
        

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
        
        # Eureka refinement
        # The refine record including:
        # refine_record = [[{"ckpt": None, "max_con_successes": None, "tb_path": None} for _ in range(sample)] for _ in range(iteration)]
        refine_record = [list() for _ in range(iteration)]
        # Init prompts
        env_cfg_dict = load_env_cfg(env_cfg_path)
        # Init LLM agents
        llm_agent = EurekaAgent(task_description, env_cfg_dict, agent_config=refine_config.get('agent', {}))
        # Run refinement loop
        # for i in range(iteration):
        i = 0
        while i < iteration:
            print(f"Refinement iteration {i+1}/{iteration}")
            for respond_id in range(sample):
                # Sample new reward functions
                reward_func, raw_response = llm_agent.func_gen()
                # clear the worksapce
                subprocess.run(["git", "checkout", "."], cwd="/home/lee/code/isaactasks")
                print(f"  Sample {respond_id+1}/{sample}: Generated new reward function.")
                # Write the new reward function to the environment config path
                rewardrules = r'@torch\.jit\.script\s*\n*def\s+compute_rewards\s*\([^)]*\).*?return\s+total_reward, reward_components'
                write_code_to_file(reward_func, env_cfg_path, rewardrules)
                # train the new reward function in sim
                log_path, max_con_successes, tb_path = train_command(whichpython=whichpython, silence=True, log_name=f"iter{i+1}_sample{respond_id+1}")
                # Check the successfulness
                if log_path is None:
                    print(f"  Sample {respond_id}: Training failed, skipping...")
                    summary_path = ""
                
                # record the result
                summary_path = os.path.join(log_path, "training_record", "training_summary.txt")
                refine_record[i].append({
                    'ckpt': log_path,
                    'max_con_successes': max_con_successes,
                    'tb_path': tb_path,
                    'prompt': llm_agent.messages,
                    'reward_func': reward_func,
                    "responses_content": raw_response,
                    "feedback_path": summary_path
                })

            # Find the best max_con_successes in this iteration
            best_idx = max(range(sample), key=lambda idx: refine_record[i][idx]['max_con_successes'])
            # Print the best result
            best_success = refine_record[i][best_idx]['max_con_successes']
            print(f"  Best consecutive successes in iteration {i+1}: {best_success} (sample {best_idx})")
            
            # If all the samples are failed, then re-run it for this iteration
            if all(record['ckpt'] is None for record in refine_record[i]):
                print(f"  All samples failed in iteration {i+1}. Re-running this iteration...")
                refine_record[i] = list()
                continue
            else:
                i += 1  # Only increment if at least one sample succeeded

            # Adding the feedback into the llm_agent
            if i < iteration:  # No need to prepare prompts for the next iteration if this was the last one
                llm_agent.add_feedback(refine_record[i-1][best_idx])

if __name__ == "__main__":
    main()