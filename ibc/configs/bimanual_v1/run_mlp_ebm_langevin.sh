#!/bin/bash

## Use "N" of the N-d environment as the arg
### --gin_bindings="ParticleEnv.n_dim=$1" \   --video

python3 ibc/ibc/path_train.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/bimanual_v1/mlp_ebm_langevin.gin \
  --task=bimanual_v1 \
  --tag=bimanual_v1 \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/bimanual/bimanual_v1_*.tfrecord'" \

