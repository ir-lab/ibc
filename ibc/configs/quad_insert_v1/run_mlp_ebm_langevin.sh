#!/bin/bash

## Use "N" of the N-d environment as the arg
### --gin_bindings="ParticleEnv.n_dim=$1" \   --video

python3 ibc/ibc/mujoco_train.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/quad_insert_v1/mlp_ebm_langevin.gin \
  --task=quad_insert_v1 \
  --tag=quad_insert_v1 \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/quad_insert_v1/quad_insert_v1_force2*.tfrecord'" \

