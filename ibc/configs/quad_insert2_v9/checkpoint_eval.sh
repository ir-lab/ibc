#!/bin/bash

## 

python3 ibc/ibc/checkpoint_eval.py -- \
 --alsologtostderr \
 --num_episodes=1 \
 --task=quad_insert2_v11 \
 --saved_model_path=/home/docker/irl_control_container/data/ibc_eval/quad_insert2_v11/2023-06-29_08:18:06/policies/greedy_policy \
 --checkpoint_path=/home/docker/irl_control_container/data/ibc_eval/quad_insert2_v11/2023-06-29_08:18:06/policies/checkpoints/policy_checkpoint_0000012000
