#!/bin/bash

# List of taskconfigs to run
taskconfigs=(
    "/home/lee/code/isaactasks/shadow_hand/taskconfig.yaml"
    "/home/lee/code/isaactasks/allegro_hand/taskconfig.yaml"
    "/home/lee/code/isaactasks/humanoid_amp/taskconfig.yaml"
    "/home/lee/code/isaactasks/humanoid/taskconfig.yaml"
    "/home/lee/code/isaactasks/anymal/taskconfig.yaml"
    "/home/lee/code/isaactasks/cartpole/taskconfig.yaml"
    "/home/lee/code/isaactasks/quadcopter/taskconfig.yaml"
    "/home/lee/code/isaactasks/ingenuity/taskconfig.yaml"
    "/home/lee/code/isaactasks/franka_cabinet/taskconfig.yaml"
    "/home/lee/code/isaactasks/ant/taskconfig.yaml"
)

# Loop through each taskconfig and run the command
for config in "${taskconfigs[@]}"
do
    echo "Running with config: $config"
    python main.py --simtrain --taskconfig "$config"
    echo "Completed run with config: $config"
    echo "----------------------------------------"
done

echo "All simtrain runs completed!"