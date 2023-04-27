#!/bin/bash

## 

python3 ibc/data/car_policy.py -- \
 --alsologtostderr \
 --num_episodes=2 \
 --policy=random \
 --task=quad_insert_v1 \
 --dataset_path=ibc/data/quad_insert_v1/2d_oracle_particle.tfrecord \
 --replicas=1  \
 --use_image_obs=False
