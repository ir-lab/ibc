#!/bin/bash

## Use "N" of the N-d environment as the arg
### --gin_bindings="ParticleEnv.n_dim=$1" \   --video

python3 ibc/ibc/path_train.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/insert_v1/mlp_ebm_langevin.gin \
  --task=insert_v1 \
  --tag=insert_v1 \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/insert_v1/bimanual_insert_v1*.tfrecord'" \

