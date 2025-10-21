"""
LLM Agent Module

Implements the LLM-based agent for autonomous simulation refinement.
Handles communication with various LLM APIs and prompt engineering.
"""
import re
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.refinement.files_operation import load_prompts

class EurekaAgent():
    def __init__(self, task_description, env_cfg_dict: dict, agent_config: dict):
        self.prompts_dict = load_prompts()
        self.task_description = task_description
        self.env_cfg_dict = env_cfg_dict
        self.model = agent_config.get('model')
        self.base_url = agent_config.get('base_url')
        # load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        # init messages
        system_content, user_content = self.init_prompt()
        self.messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]


    def init_prompt(self):
        # system prompt
        self.code_output_tip = self.prompts_dict["code_output_tip"]
        reward_template = self.env_cfg_dict["reward_code"]
        initial_system = self.prompts_dict["initial_system"]
        # feedback
        self.code_feedback = self.prompts_dict["code_feedback"]
        self.execution_error_feedback = self.prompts_dict["execution_error_feedback"]
        self.policy_feedback = self.prompts_dict["policy_feedback"]

        system_content = initial_system.format(task_reward_template=reward_template) + self.code_output_tip
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
        # task_obs_code_string = (
        #     self.env_cfg_dict["observation_code"]
        #     + self.env_cfg_dict["intermediate_code"]
        # )
        task_obs_code_string = self.env_cfg_dict["env_code"]
        user_content= initial_user.format(task_obs_code_string=task_obs_code_string, task_description=self.task_description)
        return system_content, user_content
    
    def add_feedback(self, feedback: dict):
        # refine_record[i].append({
        #     'ckpt': log_path,
        #     'max_con_successes': max_con_successes,
        #     'tb_path': tb_path,
        #     'prompt': llm_agent.messages,
        #     'reward_func': reward_func,
        #     "responses_content": raw_response,
        #     "feedback_path": os.path.join(log_path, "training_record", "training_summary.txt")
        # })
        if feedback["ckpt"] is None:
            feedback_content = self.execution_error_feedback(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")

        else:
            # TODO: How to write the reward components?
            feedback_content = self.policy_feedback
            with open(feedback["feedback_path"], "r") as f:
                feedback_content += f.read()
            feedback_content += self.code_feedback

        feedback_content += self.code_output_tip

        # Add feedback message to the conversation history
        if len(self.messages) == 2:
            self.messages += [{"role": "assistant", "content": feedback["responses_content"]}]
            self.messages += [{"role": "user", "content": feedback_content}]
        else:
            assert len(self.messages) == 4
            self.messages[-2] = {"role": "assistant", "content": feedback["responses_content"]}
            self.messages[-1] = {"role": "user", "content": feedback_content}
    

    def func_gen(self) -> str:
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # sending the messaages to the LLM API and get the response
                completion = self.client.chat.completions.create(
                    extra_headers={},
                    extra_body={},
                    model=self.model,
                    temperature=0.8,
                    n=1,
                    messages=self.messages
                )
                response = completion.choices[0].message.content
                # filtering the response to extract only the python function code
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
                        if re.search(r'@torch\.jit\.script\s*\n*def\s+compute_rewards\s*\([^)]*\).*?return\s+total_reward, reward_components', code_blocks, re.DOTALL):
                            return code_blocks, response
                
                # If we get here, no valid code block was found
                retry_count += 1
            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")
            
            # If we've exhausted all retries
        raise RuntimeError("Failed to generate valid reward function code after 10 attempts.")