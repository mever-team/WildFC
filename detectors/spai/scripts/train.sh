#!/bin/bash

python -m spai train \
  --cfg "./configs/spai.yaml" \
  --batch-size 24 \
  --pretrained "ckpt/pretrained_model.pth" \
  --output "./outputs/train_run" \
  --data-path "./data/train.csv" \
  --tag "wildfc_spai" \
  --amp-opt-level "O1" \
  --data-workers 8 \
  --save-all \
  --opt "DATA.VAL_BATCH_SIZE" "104" \
  --opt "DATA.TEST_BATCH_SIZE" "4" \
  --opt "MODEL.FEATURE_EXTRACTION_BATCH" "128" \
  --opt "DATA.VAL_PREFETCH_FACTOR" "1" \
  --opt "DATA.TEST_PREFETCH_FACTOR" "1" \
  --opt "AUG.WEBP_COMPRESSION_PROB" "0.5"
