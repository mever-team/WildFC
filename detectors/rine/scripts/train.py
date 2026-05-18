from src.utils import get_transforms, train_one_experiment,get_our_trained_model, evaluation
import os
import argparse
from datetime import datetime
from src.convert_to_onnx import export_onnx_model
from src.data import EvaluationDatasetFromCSV
import glob
from collections import defaultdict
import pandas as pd
from torch.utils.data import DataLoader
parser = argparse.ArgumentParser()
parser.add_argument("--train_csv_path", required=True, help="Path to the training CSV file")
parser.add_argument("--save_path", required=True, help="Path to save the checkpoint")
parser.add_argument(
    "--eval_folder_path",
    type=str,
    required=True,
    help="Path to evaluation folder containing CSV files"
)
parser.add_argument("--pretrained_path",required=True,help="path to the pretrained weight path")
args = parser.parse_args()
model_name = "online_rine_mever"
device = "cuda"
workers = 4
backbone = 'Dinov2'
processing_method = 'texturecrop'
pretrained_path=args.pretrained_path
train_csv_path = args.train_csv_path
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
eval_csvs = sorted(glob.glob(os.path.join(args.eval_folder_path, "*.csv")))
print(f"Found {len(eval_csvs)} evaluation CSV files in {args.eval_folder_path}")
# TRAIN
experiment = {
    "training_set": "online_sid_mever",
    "training_set_csv": train_csv_path,
    "backbone": (backbone, 1024),
    "factor": 0.8,
    "nproj": 1,
    "proj_dim": 512,
    "batch_size": 16,
    "lr": 1e-3,
    "pretrained": pretrained_path,
    "save_ckpt_path": save_path,
}

transforms_train, transforms_test = get_transforms(
        backbone=backbone,
        processing_method=processing_method,
        augmentations=True)

train_one_experiment(
        experiment=experiment,
        model_name= model_name,
        epochss=[1],
        epochs_reduce_lr=[6],
        transforms_train=transforms_train,
        transforms_val=transforms_train,
        transforms_test=transforms_test,
        workers=workers,
        device=device,
        without=None,
        store=True,
    )

## Onnx export
model_path = os.path.join(save_path, f"{model_name}.pth")
output_path = os.path.join(save_path, f"{model_name}.onnx")
export_onnx_model(model_path=model_path, output_path=output_path)

model_path = os.path.join(save_path, f"{model_name}.pth")

# Evaluation
generator_to_csvs = {}
for csv_file in eval_csvs:
    df = pd.read_csv(csv_file)
    if df.empty:
        continue
    generator_name = os.path.splitext(os.path.basename(csv_file))[0]
    generator_to_csvs[generator_name] = csv_file
print(f'Found {len(generator_to_csvs)} generators for evaluation: {list(generator_to_csvs.keys())}')
_, transforms = get_transforms(backbone=backbone, processing_method=processing_method)
test = [
    (
        generator_name,
        DataLoader(
            EvaluationDatasetFromCSV(csv_file=csv_file, transforms=transforms, perturb=None),
            batch_size=8,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        ),
    )
    for generator_name, csv_file in generator_to_csvs.items()
]
model = get_our_trained_model(bcb=backbone, model_name=model_path, device=device)
model.to(device)
metrics_csv = os.path.join(save_path, f"metrics.csv")
predictions_csv = os.path.join(save_path, f"predictions.csv")
evaluation(model, test, device, processing_method=processing_method, ours=True, metrics_csv=metrics_csv,predictions_csv=predictions_csv,eval_csvs=eval_csvs)
