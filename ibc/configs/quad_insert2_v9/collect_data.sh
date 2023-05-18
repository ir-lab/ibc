#!/bin/bash

## 

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=1 \
 --policy=random \
 --task=quad_insert2_v9 \
 --dataset_path=ibc/data/quad_insert2_v9/2d_oracle.tfrecord \
 --replicas=1  \
 --use_image_obs=False
