# Isaac-Sim2Real-Pipeline

We aim to construct a comprehensive toolkit designed to automate the end-to-end pipeline for sim-to-real reinforcement learning. This system will automatically train specified RL tasks within the Isaac Lab simulator, utilise an LLM-based agent for iterative performance optimisation, and subsequently facilitate the seamless migration of the trained policy to a physical environment.

Core Features:

- Interaction friendly
- Multi-agent support
- Multi-algorithm support

## STEP 1:

In the Prototype v1 stage, we simply call a mature and off-the-shelf Isaaclab project, either customised to fit a specific task or auto-generated via the Isaac command.

- The output: A decent simulation can be recorded (including ckpt, comprehensive videos and quantitative results)

(**Optional**) Manually setting up the Isaaclab environment is still a labour-intensive task, which involves a human expert to design. Our further work will concentrate on automating the simulation setup:


- The physical layout, which contains physical rules and each interactive object
- The Reward
- observations, action spaces for the tasks, which should align with the real-world setting
- a proper event, terminations for the real-world randomisation
- commands for defining the goal

## STEP 2: Automated Simulation Refinement

This stage implements a recursive pipeline leveraging an LLM agent to autonomously refine simulation results using performance feedback. The process is divided into two phases:

1. **Baseline Implementation:** First, we will integrate the foundational Eureka framework into our toolkit to establish a performance baseline.
2. **Advanced Optimisation:** Subsequently, we will develop an enhanced methodology designed to systematically improve upon the Eureka outputs and validate the performance gains.

## STEP 3: Overcoming the GAP between Sim2Real

After setting up the robotic environment in the real world, the previously trained model ckpt probably will not function directly in the real-world setting. Which means some sim2real approaches should be accepted in this stage to fill the gap.

There are two implementations we should try on our prototype:

1. DrEuleka: Using LLMs to design the domain randomisation for robustness.
2. Our approach (v1): Using the videos recorded from both the simulation and the real world as a clue to generate an opinion for improving the environmental setting on the Isaaclab simulation.