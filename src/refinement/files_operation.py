import os
import re



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


