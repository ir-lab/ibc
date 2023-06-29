#!/bin/bash

## 

python3 ibc/ibc/checkpoint_eval.py -- \
 --alsologtostderr \
 --num_episodes=1 \
 --task=quad_insert2_v11 \
 --saved_model_path=ibc/ibc/configs/quad_insert2_v9/2023-05-30_08:14:15/policies/greedy_policy \
 --checkpoint_path=ibc/ibc/configs/quad_insert2_v9/2023-05-30_08:14:15/policies/checkpoints/policy_checkpoint_0000012000
