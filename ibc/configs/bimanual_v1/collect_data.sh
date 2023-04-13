#!/bin/bash

## 

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=1 \
 --policy=random \
 --task=dual_insert_v3 \
 --dataset_path=ibc/data/dual_insert_v3/2d_oracle_particle.tfrecord \
 --replicas=1  \
 --use_image_obs=False
