
python -m spai test \
  --cfg "./configs/wildfc_spai.yaml" \
  --batch-size 4 \
  --model "ckpt/model.pth" \
  --output "./outputs/eval_run" \
  --tag "wildfc_spai" \
  --opt "MODEL.PATCH_VIT.MINIMUM_PATCHES" "4" \
  --opt "DATA.NUM_WORKERS" "8" \
  --opt "MODEL.FEATURE_EXTRACTION_BATCH" "400" \
  --opt "DATA.TEST_PREFETCH_FACTOR" "1" \
  --update-csv \
  --test-csv "./data/test_set_1.csv" \
  --test-csv "./data/test_set_2.csv" \
  --test-csv-root-dir "./data/eval_sets"