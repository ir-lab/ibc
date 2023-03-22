#!/bin/bash

## Use "N" of the N-d environment as the arg
### --gin_bindings="ParticleEnv.n_dim=$1" \   --video

python3 ibc/ibc/path_train.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/bimanual_single_quat/mlp_ebm_langevin.gin \
  --task=bimanual_single_quat_v2 \
  --tag=bimanual_single_quat_v2 \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/bimanual_single_quat/bimanual_quat*.tfrecord'" \

