"""
LLM Agent Module

Implements the LLM-based agent for autonomous simulation refinement.
Handles communication with various LLM APIs and prompt engineering.
"""

from openai import OpenAI
from src.refinement.files_operation import load_prompts

class EurekaAgent():
    def __init__(self, task_description, env_cfg_dict: dict, agent_config: dict):
        self.prompts_dict = load_prompts()
        self.task_description = task_description
        self.env_cfg_dict = env_cfg_dict
        self.model = agent_config.get('model')
        self.base_url = agent_config.get('base_url')
        self.api_key = agent_config.get('api_key')
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        # init messages
        system_content, user_content = self.init_prompt()
        self.messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]


    def init_prompt(self):
        # system prompt
        code_output_tip = self.prompts_dict["code_output_tip"]
        reward_template = self.env_cfg_dict["reward_code"]
        initial_system = self.prompts_dict["initial_system"]
        system_content = initial_system.format(task_reward_template=reward_template) + code_output_tip
        # user prompt
        # task_dict = {
        #     "code_output_tip": code_output_tip,
        #     "initial_system": initial_system,
        #     "initial_user": initial_user,
        #     "reward_signature": reward_signature,
        #     "observation_code": observation_code,
        #     "reward_code": reward_code,
        #     "intermediate_code": intermediate_code,
        #     "env_cfg_code": env_cfg_code,
        #     "env_file_path": env_cfg_path
        # }
        initial_user = self.prompts_dict["initial_user"]
        task_obs_code_string = (
            self.env_cfg_dict["observation_code"]
            + self.env_cfg_dict["intermediate_code"]
        )
        user_content= initial_user.format(task_obs_code_string=task_obs_code_string, task_description=self.task_description)
        return system_content, user_content
    
    # def add_feedback(self, feedback: str):
    #     pass

    def func_gen(self) -> str:
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # sending the messaages to the LLM API and get the response
                completion = self.client.chat.completions.create(
                    extra_headers={},
                    extra_body={},
                    model=self.model or "x-ai/grok-code-fast-1",
                    messages=self.messages
                )
                response = completion.choices[0].message.content
                # filtering the response to extract only the python function code
                import re
                patterns = [
                    r'```python(.*?)```',
                    r'```(.*?)```',
                    r'"""(.*?)"""',
                    r'""(.*?)""',
                    r'"(.*?)"',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, response, re.DOTALL)
                    if matches:
                        code_blocks = matches[0].strip()
                        if re.search(r'@torch\.jit\.script\s*\n*def\s+compute_rewards\s*\([^)]*\).*?return\s+total_reward', code_blocks, re.DOTALL):
                            return code_blocks
                
                # If we get here, no valid code block was found
                retry_count += 1
            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")
            
            # If we've exhausted all retries
            return "ERROR: Failed to generate valid code after 10 attempts. The LLM response did not contain a properly formatted compute_rewards function."