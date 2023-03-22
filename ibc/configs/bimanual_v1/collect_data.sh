#!/bin/bash

## 

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=1 \
 --policy=random \
 --task=bimanual_single_sixdof_v2 \
 --dataset_path=ibc/data/bimanual_single_sixdof/2d_oracle_particle.tfrecord \
 --replicas=1  \
 --use_image_obs=False
