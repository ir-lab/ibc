#!/bin/bash

## Use "N" of the N-d environment as the arg
### --gin_bindings="ParticleEnv.n_dim=$1" \   --video

python3 ibc/ibc/path_train.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/particle/mlp_ebm_langevin.gin \
  --task=PARTICLE \
  --tag=particle_3d \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/particle_3d/2d_oracle_particle*.tfrecord'" \

