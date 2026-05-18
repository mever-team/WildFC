This project uses and adapts code from the following work:

- *SPAI: Spectral AI-Generated Image Detector*
- Authors: Dimitrios Karageorgiou, Symeon Papadopoulos, Ioannis Kompatsiaris, Efstratios Gavves
- Repository: [SPAI GitHub Repository](https://github.com/mever-team/spai)
- Paper: [CVPR 2025 Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html)

![Overview of the SPAI architecture](detectors/spai/docs/overview.png)

## Installation

### Required Libraries

To train and evaluate SPAI, create a Conda environment and install the required dependencies as follows:

```bash
conda create -n spai python=3.11
conda activate spai
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Pre-trained Weights

Pre-trained weights are available here:

[Download Pre-trained Weights](https://itigr-my.sharepoint.com/:f:/g/personal/apantsios_iti_gr/IgAbMc4s2ETARaCokbAQEZGqAU9usxzh43MG-psl06b6gE4?e=h8FbVN)

Place the downloaded weights inside the `ckpt/` folder.

---

## Data Format

Training and evaluation datasets require a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `filepath` | string | Absolute or relative path to the image file |
| `class` | integer | Label: `0` for real images, `1` for fake images |
| `split` | string | Dataset split identifier (`"train"` or `"test"`) |


**Training CSV:** A single CSV file containing all training samples with `filepath`, `class`, and `split="train"`.

**Evaluation CSVs:** One or more CSV files stored in a folder, where each file represents a separate evaluation set. The filename is used as the evaluation set name in the results.

---

## Training

Set:
- `--data-path` to the training CSV file,
- `--pretrained` to the pre-trained model checkpoint,
- `--output` to the directory where checkpoints and logs will be saved.

Training can then be performed as follows:

```bash
python -m spai train \
  --cfg "./configs/spai.yaml" \
  --batch-size 24 \
  --pretrained "ckpt/pretrained_model.pth" \
  --output "/save_dir" \
  --data-path "train_csv_path" \
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
```

---

## Evaluation

After training a model, evaluation can be performed as follows:

```bash
python -m spai test \
  --cfg "./configs/wildfc_spai.yaml" \
  --batch-size 4 \
  --model "ckpt/model.pth" \
  --output save_dir \
  --tag "wildfc_spai" \
  --opt "MODEL.PATCH_VIT.MINIMUM_PATCHES" "4" \
  --opt "DATA.NUM_WORKERS" "8" \
  --opt "MODEL.FEATURE_EXTRACTION_BATCH" "400" \
  --opt "DATA.TEST_PREFETCH_FACTOR" "1" \
  --update-csv \
  --test-csv "<test_csv_path_1>" \
  --test-csv "<test_csv_path_2>" \
  --test-csv-root-dir "<test_folder_path>"
```

### Arguments

- `--test-csv`: Path to a CSV file containing the evaluation image paths and labels.
- `--test-csv-root-dir`: Directory containing multiple evaluation CSV files.

Multiple evaluation datasets can be placed inside the `--test-csv-root-dir` folder and evaluated together by providing multiple `--test-csv` arguments.