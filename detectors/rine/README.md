This project uses and adapts code from the following work:

- *Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection*
- Authors: Christos Koutlis, Symeon Papadopoulos
- Repository: https://github.com/mever-team/rine
- Paper: https://arxiv.org/abs/2402.19091

## Setup
Clone the repository:
```
git clone https://github.com/mever-team/WildFC/detectors/rine
```
Create the environment:
```
conda create -n rine python=3.9
conda activate rine
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
Pre-trained weights are provided here: link https://itigr-my.sharepoint.com/:f:/g/personal/apantsios_iti_gr/IgAbMc4s2ETARaCokbAQEZGqAU9usxzh43MG-psl06b6gE4?e=h8FbVN and should be place in ckpt/ folder.

## Data Format

The training and evaluation datasets requires a CSV file with the following columns:


| Column | Type | Description |
|--------|------|-------------|
| `filepath` | string | Absolute or relative path to the image file |
| `class` | integer | Label: `0` for real, `1` for fake |
| `split` | string | Data split identifier (`"train"` or `"test"`)|

**Training CSV:** A single CSV file with columns `filepath` and `class` and `split = "train"` containing all training images.

**Evaluation CSV:** One or more CSV files placed in a folder. Each file represents one evaluation set. Filename becomes the 'generator' name in results.

## Training
Train using the training script with required arguments:

```bash
python scripts/train.py \
    --train_csv_path /path/to/training.csv \
    --save_path /path/to/output/checkpoint \
    --eval_folder_path /path/to/evaluation/csvs \
    --pretrained_path /path/to/pretrained/dinov2.pth
```

**Arguments:**
- `--train_csv_path`: Path to a single CSV file with training data
- `--save_path`: Directory where model checkpoint and results will be saved
- `--eval_folder_path`: Directory containing one or more CSV files for evaluation
- `--pretrained_path`: Path to pretrained weights

## Evaluation
Evaluate a pre-trained model on test datasets:

```bash
python scripts/eval.py \
    --model_path /path/to/model.pth \
    --save_path /path/to/output/results \
    --eval_folder_path /path/to/evaluation/csvs
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint (.pth file)
- `--save_path`: Directory where evaluation results will be saved
- `--eval_folder_path`: Directory containing CSV files for evaluation




