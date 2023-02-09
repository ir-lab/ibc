#!/bin/bash

python3 ibc/ibc/path_train.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/pushing_states/mlp_ebm.gin \
  --task=PUSH \
  --tag=pushing_states_traget \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/block_push_states_location_target/oracle_push*.tfrecord'" \
  --video
