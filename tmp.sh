#!/usr/bin/env bash

./jedi.sh run_fsl --fsl_python_arguments=""


--bypass_sql
--network_type=weightedlinear
--agent_type=experiencemodel
--environment_type=maze
--phi_frames=1
-max_frames_per_episode=100
--threads=12
--copy_frequency=10
--iterations_before_training=3000

