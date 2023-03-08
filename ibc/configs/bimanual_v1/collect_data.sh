#!/bin/bash

## 

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=4 \
 --policy=random \
 --task=bimanual_v1 \
 --dataset_path=ibc/data/bimanual/2d_oracle_particle.tfrecord \
 --replicas=1  \
 --use_image_obs=False
