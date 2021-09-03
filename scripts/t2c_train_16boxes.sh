#!/usr/bin/env bash

python ./scripts/launch.py \
    --nproc_per_node "$1" "talk2car/train_end2end.py" \
     --cfg "cfgs/large_deeper_16boxes.yaml" --model-dir "trained_models_16boxes"