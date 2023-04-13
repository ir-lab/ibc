#!/bin/bash

## Use "N" of the N-d environment as the arg
### --gin_bindings="ParticleEnv.n_dim=$1" \   --video

python3 ibc/ibc/path_train.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/dual_insert_v3/mlp_ebm_langevin.gin \
  --task=dual_insert_v3 \
  --tag=dual_insert_v3 \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/dual_insert_v3/dual_insert_v3_quat.tfrecord'" \

