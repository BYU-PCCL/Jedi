#!/usr/bin/env bash

./jedi.sh fsl_copy

for i in `seq 1 15`; # 1 15 when ready
    do
        # ./jedi.sh run_fsl --fsl_username=el3ment --fsl_python_arguments="--name=state_weighted --network_type=weightedlinear --agent_type=experiencemodel --environment_type=maze --phi_frames=1 --max_frames_per_episode=100 --threads=12 --copy_frequency=10 --iterations_before_training=3000" &
        ./jedi.sh run_fsl --fsl_username=el3ment --fsl_python_arguments="--name=superchargetest --deterministic --network_type=linear --agent_type=experiencemodel --environment_type=maze --phi_frames=1 --max_frames_per_episode=100 --threads=12 --copy_frequency=10 --iterations_before_training=3000" &
    done
wait
echo "Done"