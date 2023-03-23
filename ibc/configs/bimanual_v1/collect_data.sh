#!/bin/bash

## 

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=1 \
 --policy=random \
 --task=insert_v1_sixdof \
 --dataset_path=ibc/data/insert_v1_sixdof/2d_oracle_particle.tfrecord \
 --replicas=1  \
 --use_image_obs=False
