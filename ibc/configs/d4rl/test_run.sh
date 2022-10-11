#!/bin/bash

# Use name of d4rl env as first arg

CMD='python3 ibc/ibc/train_test.py '
GIN='ibc/ibc/configs/d4rl/mlp_ebm_test.gin'
#GIN='ibc/ibc/configs/d4rl/test_run.gin'
DATA="train_eval.dataset_path='ibc/data/d4rl/car2/*.tfrecord'"

$CMD -- \
  --alsologtostderr \
  --gin_file=$GIN \
  --task=$1 \
  --tag=ibc_car \
  --add_time=True \
  --gin_bindings=$DATA
  # not currently calling --video because rendering is broken in the docker?r
