import os
import re
import glob
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing import event_accumulator



def load_prompts():
    # Reading configuration files from agent_config directory
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "refinement", "agent_config")
    code_output_tip_path = os.path.join(config_dir, "code_output_tip.txt")
    initial_system_path = os.path.join(config_dir, "initial_system.txt")
    initial_user_path = os.path.join(config_dir, "initial_user.txt")
    
    # Read the configuration files
    try:
        with open(code_output_tip_path, 'r') as f:
            code_output_tip = f.read()
        with open(initial_system_path, 'r') as f:
            initial_system = f.read()
        with open(initial_user_path, 'r') as f:
            initial_user = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {e}")
    
    return {
        "code_output_tip": code_output_tip,
        "initial_system": initial_system,
        "initial_user": initial_user,
    }


def load_env_cfg(env_cfg_path):
    # read the env config from env_cfg_path which is a python scripts containing useful infos for the isaaclab simultions, 
    # and copy observation (_get_observations) and related codes as a strings
    
    # Read the environment config file
    try:
        with open(env_cfg_path, 'r') as f:
            env_config_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Environment config file not found: {env_cfg_path}")
    
    # Extract the observation method (_get_observations) to return observations
    observation_code = ""
    observation_pattern = re.compile(r'def\s+_get_observations\s*\([^)]*\).*?return\s+observations', re.DOTALL)
    match = observation_pattern.search(env_config_content)
    if match:
        observation_code = match.group(0)
    else:
        raise ValueError("Could not find the _get_observations method in the environment config file.")
    
    # Extract reward calculation code - from @torch.jit.script def compute_rewards to return total_reward
    reward_code = ""
    compute_rewards_pattern = re.compile(
        r'@torch\.jit\.script\s*\n*def\s+compute_rewards\s*\([^)]*\).*?return\s+total_reward', re.DOTALL
    )
    match = compute_rewards_pattern.search(env_config_content)
    if match:
        reward_code = match.group(0)
    else:
        raise ValueError("Could not find the compute_rewards function in the environment config file.")

    # Extract compute_intermediate_values function - from @torch.jit.script to its return values
    intermediate_code = ""
    intermediate_pattern = re.compile(
        r'@torch\.jit\.script\s*\n*def\s+compute_intermediate_values\s*\([^)]*\).*?return\s+\(\s*[^)]*\s*\)', re.DOTALL
    )
    match = intermediate_pattern.search(env_config_content)
    if match:
        intermediate_code = match.group(0)
    else:
        raise ValueError("Could not find the compute_intermediate_values function in the environment config file.")
    
    # Combine all extracted information
    env_cfg_dict = {
        "observation_code": observation_code,
        "reward_code": reward_code,
        "intermediate_code": intermediate_code,
    }
    return env_cfg_dict




def write_code_to_file(func_strings: str, target_script: str, rules: str):
    # Read the target script file
    try:
        with open(target_script, 'r') as f:
            script_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Target script file not found: {target_script}")

    # Use the provided regex rules to find the function to replace
    pattern = re.compile(rules, re.DOTALL)
    if not pattern.search(script_content):
        raise ValueError("Could not find the target function in the target script file using the provided rules.")

    # Replace the matched function with the new func_strings
    new_script_content = pattern.sub(func_strings, script_content, count=1)

    # Write the modified content back to the file
    with open(target_script, 'w') as f:
        f.write(new_script_content)


def read_tb(tb_file, tag):
    # import time
    # time.sleep(2)
    # Path to the TensorBoard file
    if not os.path.exists(tb_file):
        print(f"Error: TensorBoard file not found: {tb_file}")
        consecutive_successes_events = None
    else:
        print(f"Reading metrics from: {tb_file}")
        # Load the event file
        ea = event_accumulator.EventAccumulator(tb_file)
        ea.Reload()
        # Extract the consecutive_successes metric if available
        if tag in ea.scalars.Keys():
            consecutive_successes_events = ea.Scalars(tag)
            if consecutive_successes_events:
                return consecutive_successes_events
                # Get the last recorded value
        else:
            print(f"Warning: {tag} metric not found in TensorBoard logs")
    return consecutive_successes_events


def get_latest_checkpoint_dir(logs_path):
    # Find all timestamp directories
    timestamp_dirs = glob.glob(f"{logs_path}/*")
    
    # Filter for directories only and sort by creation time (newest first)
    timestamp_dirs = [d for d in timestamp_dirs if os.path.isdir(d)]
    timestamp_dirs.sort(key=os.path.getctime, reverse=True)
    
    if not timestamp_dirs:
        return None
        
    # Get the newest directory and add the nn subfolder
    return timestamp_dirs[0] if os.path.exists(timestamp_dirs[0]) else None



def summarize_tensorboard(event_file_path, output_txt_path):
    """
    Reads a TensorBoard event file, summarizes the scalar data, and writes
    a human-readable summary to a text file for LLM analysis.

    Args:
        event_file_path (str): The path to the .tfevents file.
        output_txt_path (str): The path where the output .txt summary will be saved.
    """
    # 1. --- Load the Event File ---
    # Initialize an EventAccumulator and load the event file.
    # The size_guidance is set to 0 to load all data.
    size_guidance = {
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
        }

    acc = event_accumulator.EventAccumulator(event_file_path,
                                             size_guidance=size_guidance)
    acc.Reload() # This loads the data from the file.

    # 2. --- Extract and Process Scalar Data ---
    # Get a list of all scalar tags (e.g., 'rollout/ep_rew_mean', 'train/loss')
    scalar_tags = acc.Tags()['scalars']
    
    summary_lines = []
    summary_lines.append("## Reinforcement Learning Model Performance Summary\n")
    summary_lines.append(f"Source File: {os.path.basename(event_file_path)}\n")
    summary_lines.append("-" * 40 + "\n")

    for tag in scalar_tags:
        # Get all scalar events for the current tag
        events = acc.Scalars(tag)
        
        # Extract the steps and values into numpy arrays for easy processing
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        if len(values) == 0:
            continue

        # 3. --- Calculate Summary Statistics ---
        mean_val = np.mean(values)
        std_val = np.std(values)
        max_val = np.max(values)
        min_val = np.min(values)
        
        # 4. --- Calculate Trend Data ---
        # Get values at the start, middle, and end of training
        initial_idx = int(len(values) * 0.1)
        mid_idx = int(len(values) * 0.5)
        final_idx = -1 # Last element
        
        initial_perf = np.mean(values[:initial_idx]) if initial_idx > 0 else values[0]
        mid_perf = values[mid_idx]
        final_perf = np.mean(values[-initial_idx:]) if initial_idx > 0 else values[-1]

        # 5. --- Format the Output ---
        summary_lines.append(f"## Metric: {tag}\n")
        summary_lines.append(f"- **Overall Statistics:**")
        summary_lines.append(f"  - Mean: {mean_val:.4f}")
        summary_lines.append(f"  - Std Dev: {std_val:.4f} (Measures stability/variance)")
        summary_lines.append(f"  - Max Value: {max_val:.4f}")
        summary_lines.append(f"  - Min Value: {min_val:.4f}\n")
        
        summary_lines.append(f"- **Performance Trend:**")
        summary_lines.append(f"  - Initial Performance (first 10%): ~{initial_perf:.4f}")
        summary_lines.append(f"  - Mid-Training Performance (at 50%): ~{mid_perf:.4f}")
        summary_lines.append(f"  - Final Performance (last 10%): ~{final_perf:.4f}\n")
        summary_lines.append("-" * 40 + "\n")

    # 6. --- Write to File ---
    with open(output_txt_path, 'w') as f:
        f.write("\n".join(summary_lines))
        
    print(f"✅ Summary successfully written to {output_txt_path}")

# --- Example Usage ---
if __name__ == '__main__':
    # You need to find the actual .tfevents file. It's usually in a directory
    # like 'logs/PPO_1/' or similar.
    # For example: '/path/to/your/logs/PPO_1/events.out.tfevents.1668...'.
    
    # IMPORTANT: Replace this with the actual path to YOUR event file
    # You can find it by running 'ls -R' in your log directory.
    event_file = "/home/lee/code/isaactasks/shadow_hand/logs/rl_games/shadow_hand/2025-10-03_15-55-31/summaries/events.out.tfevents.1759460146.sibyl"
    output_file = 'training_summary.txt'
    
    # Check if the example path exists before running
    if os.path.exists(event_file):
        summarize_tensorboard(event_file, output_file)
    else:
        print("❌ Error: Please update the 'event_file' variable with the correct path to your TensorBoard file.")



# if __name__ == "__main__":
    # Use with your config
    # logs_path = "/home/lee/code/isaactasks/shadow_hand/logs/rl_games/shadow_hand"
    # checkpoint_dir = get_latest_checkpoint_dir(logs_path)
    # print(f"Latest checkpoint directory: {checkpoint_dir}")
    # Testing the read_tb function
    # file_path = "/home/lee/code/isaactasks/shadow_hand/logs/rl_games/shadow_hand/2025-10-03_15-55-31/summaries/events.out.tfevents.1759460146.sibyl"
    # tag = 'Episode/consecutive_successes'
    # consecutive_successes = read_tb(file_path, tag)