#!/bin/bash

## 

python3 ibc/ibc/checkpoint_eval.py -- \
 --alsologtostderr \
 --num_episodes=5 \
 --task=quad_insert2_v11 \
 --model_path=/home/docker/irl_control_container/data/ibc_eval/quad_insert2_v11/2023-07-09_07:44:55\
 --checkpoint=12000

