#!/bin/bash

# Use name of d4rl env as first arg

CMD='python3 ibc/ibc/path_train.py '
GIN='ibc/ibc/configs/d4rl/mlp_ebm_pathv2.gin'
#GIN='ibc/ibc/configs/d4rl/test_run.gin'
DATA="train_eval.dataset_path='ibc/data/d4rl/path_follow_v2/*.tfrecord'"

$CMD -- \
  --alsologtostderr \
  --gin_file=$GIN \
  --task=path_follow_v2 \
  --tag=path_follow_v2 \
  --add_time=True \
  --gin_bindings=$DATA
  # not currently calling --video because rendering is broken in the docker?r
