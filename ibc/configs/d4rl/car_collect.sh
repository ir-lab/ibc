#!/bin/bash


# --dataset_path=ibc/data/d4rl/car2/2d_oracle_car.tfrecord \

# --video=True \
python3 ibc/data/car_policy.py -- \
 --alsologtostderr \
 --num_episodes=1 \
 --policy=random \
 --task=CAR \
 --use_image_obs=False \
