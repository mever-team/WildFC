import logging
import os
import pathlib
import time
import datetime
from pathlib import Path
from typing import Optional, Any
import pandas as pd
import numpy as np
import csv
import neptune
import cv2
import click
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchmetrics
import yacs
from torch import nn
from torch.nn import TripletMarginLoss
from torch.utils.tensorboard import SummaryWriter
from torch.amp import  GradScaler,autocast
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter
from tqdm import tqdm
from yacs.config import CfgNode

import spai.data.data_finetune
from spai.config import get_config
from spai.models import build_cls_model
from spai.data import build_loader, build_loader_test
from spai.lr_scheduler import build_scheduler
from spai.models.sid import AttentionMask
from spai.onnx import compare_pytorch_onnx_models
from spai.optimizer import build_optimizer
from spai.logger import create_logger
from spai.utils import (
    # load_checkpoint,
    load_pretrained,
    save_checkpoint,
    get_grad_norm,
    # auto_resume_helper,
    find_pretrained_checkpoints,
    inf_nan_to_num
)
from spai.models import losses
from spai import explainability, metrics, data_utils

"""
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
"""

cv2.setNumThreads(1)
logger: Optional[logging.Logger] = None


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-size", type=int,
              help="Batch size for a single GPU.")
@click.option("--learning-rate", type=float)
@click.option("--data-path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="path to dataset")
@click.option("--csv-root-dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--pretrained",
              type=click.Path(exists=True, dir_okay=False),
              help="path to pre-trained model")
@click.option("--resume", is_flag=True,
              help="resume from checkpoint")
@click.option("--accumulation-steps", type=int, default=1,
              help="Gradient accumulation steps.")
@click.option("--use-checkpoint", is_flag=True,
              help="Whether to use gradient checkpointing to save memory.")
@click.option("--amp-opt-level", type=click.Choice(["O0", "O1", "O2"]), default="O1",
              help="mixed precision opt level, if O0, no amp is used")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--local_rank", type=int, default=0,
              help="local_rank for distributed training")
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--data-workers", type=int,
              help="Number of worker processes to be used for data loading.")
@click.option("--disable-pin-memory", is_flag=True)
@click.option("--data-prefetch-factor", type=int)
@click.option("--save-all", is_flag=True)
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
def train(
    cfg: Path,
    batch_size: Optional[int],
    learning_rate: Optional[float],
    data_path: Path,
    csv_root_dir: Optional[Path],
    lmdb_path: Optional[Path],
    pretrained: Optional[Path],
    resume: bool,
    accumulation_steps: int,
    use_checkpoint: bool,
    amp_opt_level: str,
    output: Path,
    tag: str,
    local_rank: int,
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    data_workers: Optional[int],
    disable_pin_memory: bool,
    data_prefetch_factor: Optional[int],
    save_all: bool,
    extra_options: tuple[str, str]
) -> None:
    if csv_root_dir is None:
        csv_root_dir = data_path.parent
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "data_path": str(data_path),
        "csv_root_dir": str(csv_root_dir),
        "lmdb_path": str(lmdb_path),
        "pretrained": str(pretrained) if pretrained is not None else None,
        "resume": resume,
        "accumulation_steps": accumulation_steps,
        "use_checkpoint": use_checkpoint,
        "amp_opt_level": amp_opt_level,
        "output": str(output),
        "tag": tag,
        "local_rank": local_rank,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "data_workers": data_workers,
        "disable_pin_memory": disable_pin_memory,
        "data_prefetch_factor": data_prefetch_factor,
        "opts": extra_options
    })
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(local_rank)

    use_amp = config.AMP_OPT_LEVEL != "O0"

    # Set a fixed seed to all the random number generators.
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    if config.TRAIN.SCALE_LR:
        # Linear scale the learning rate according to total batch size - may not be optimal.
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
        # Gradient accumulation also need to scale the learning rate.
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export and display current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config, logger, is_pretrain=False, is_test=False
    )
    print("Data loaders built successfully.")
    
    
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    scaler = GradScaler(enabled= use_amp)
    #if config.AMP_OPT_LEVEL != "O0":
    #    model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion: nn.Module = losses.build_loss(config)
    logger.info(f"Loss: \n{criterion}")

    # if config.TRAIN.AUTO_RESUME:
    #     resume_file = auto_resume_helper(config.OUTPUT, logger)
    #     if resume_file:
    #         if config.MODEL.RESUME:
    #             logger.warning(
    #                 f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
    #             )
    #         config.defrost()
    #         config.MODEL.RESUME = resume_file
    #         config.freeze()
    #         logger.info(f'auto resuming from {resume_file}')
    #     else:
    #         logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    # if config.MODEL.RESUME:
    #     max_accuracy = load_checkpoint(
    #         config, model_without_ddp.get_vision_transformer(), optimizer, lr_scheduler, logger)
    #     if config.TRAIN.MODE == "contrastive":
    #         acc, ap, auc, loss = validate_knn(config, data_loader_val, dataset_val, model)
    #     else:
    #         acc, ap, auc, loss = validate(config, data_loader_val, model, criterion, neptune_run)
    #     logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc:.1f}%")
    #     logger.info(f"AP of the network on the {len(dataset_val)} test images: {ap:.1f}%")
    #     logger.info(f"AUC of the network on the {len(dataset_val)} test images: {auc:.1f}%")
    #     if config.EVAL_MODE:
    #         return
    # elif config.PRETRAINED:
    if config.PRETRAINED:
        load_pretrained(config, model_without_ddp, logger) # model_without_ddp.get_vision_transformer()
    else:
        model_without_ddp.unfreeze_backbone()
        logger.info(f"No pretrained model. Backbone parameters are trainable.")

    
    # model_without_ddp.freeze_base_network()

    # if config.THROUGHPUT_MODE:
    #     throughput(data_loader_val, model, logger)
    #     return

    test_datasets_names, test_datasets, test_loaders = build_loader_test(config, logger)
    
    train_model(
        config,
        model,
        model_without_ddp,
        data_loader_train,
        data_loader_val,
        test_loaders,
        dataset_val,
        test_datasets,
        test_datasets_names,
        criterion,
        optimizer,
        scaler,
        lr_scheduler,
        log_writer,
        save_all=save_all
    )


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-size", type=int,
              help="Batch size for a single GPU.")
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--split", type=str, default="test",
              help="The data split which will be tested. Actually, this value is expected to be "
                   "present in the `split` column of the provided csv files. Only samples "
                   "in the csv belonging to the provided split will be tested.")
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--resize-to", type=int,
              help="When this argument is provided the testing images will be resized "
                   "so that their biggest dimension does not exceed this value.")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
@click.option("--verbose", is_flag=True, help="Print verbose output.")
@click.option("--update-csv", is_flag=True,
              help="When this flag is provided the predicted score for each sample is "
                   "written to the dataset csv, under a new column named as "
                   "{tag}_epoch_{epoch_num}_{crop_approach}.")
def test(
    cfg: Path,
    batch_size: Optional[int],
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    split: str,
    lmdb_path: Optional[Path],
    model: Path,
    output: Path,
    tag: str,
    resize_to: Optional[int],
    extra_options: tuple[str, str],
    verbose: bool,
    update_csv: bool
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "resize_to": resize_to,
        "opts": extra_options
    })

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())
    
    test_datasets_names, test_datasets, test_loaders = build_loader_test(config, logger,
                                                                         split=split)
    model_checkpoints: list[pathlib.Path] = find_pretrained_checkpoints(config)
    criterion = losses.build_loss(config)
    for i, model_ckpt in enumerate(model_checkpoints):
        if i == 0:
            logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        model.cuda()
        if i == 0:
            logger.info(str(model))
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Number of Params: {n_parameters}")
            if hasattr(model, "flops"):
                flops = model.flops()
                logger.info(f"Number of GFLOPs: {flops / 1e9}")

        checkpoint_epoch: int = load_pretrained(config, model, logger,
                                                checkpoint_path=model_ckpt, verbose=i==0)
        log = {}
        csv_rows = []
        accs, aps, aucs, losss = [], [], [], []
        # Test the model.
        for test_data_loader, test_dataset, test_data_name in zip(test_loaders,
                                                                  test_datasets,
                                                                  test_datasets_names):
            predictions: Optional[dict[int, tuple[float, Optional[AttentionMask]]]] = None
            if update_csv:
                acc, ap, auc, loss, predictions = validate(
                    config, test_data_loader, model, criterion, None,
                    return_predictions=True
                )
            else:
                acc, ap, auc, loss = validate(config, test_data_loader,
                                              model, criterion, None)
            logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch}")
            logger.info(f"ACC: {acc:.3f} AP: {ap:.3f} AUC: {auc:.3f} LOSS: {loss:.4f}")
        
            # Store metrics
            log[test_data_name] = {
                "acc": acc,
                "ap": ap,
                "auc": auc,
                "loss": loss,
            }

            accs.append(acc)
            aps.append(ap)
            aucs.append(auc)
            losss.append(loss)

            if predictions is not None:
                export_dir = Path(os.path.join(config.OUTPUT, f"epoch_{checkpoint_epoch}"))
                os.makedirs(export_dir, exist_ok=True)
                column_name: str = "probability"
                scores: dict[int, float] = {i: t[0] for i, t in predictions.items()}
                attention_masks: dict[int, pathlib.Path] = {
                    i: t[1].mask for i, t in predictions.items() if t[1] is not None
                }
                test_dataset.update_dataset_csv(
                    column_name, scores, export_dir=export_dir)
                
                if len(attention_masks) == len(scores):
                    test_dataset.update_dataset_csv(
                        f"{column_name}_mask", attention_masks, export_dir=export_dir
                    )
                generator_csv_path = export_dir / f"{test_data_name}.csv"
                predictions_csv_path = Path(os.path.join(output,f"epoch_{checkpoint_epoch}","predictions.csv"))
                # Read the generator CSV we just saved
                df_gen = pd.read_csv(generator_csv_path)

                # Append to total CSV if it exists, else create new
                if predictions_csv_path.exists():
                    df_total = pd.read_csv(predictions_csv_path)
                    df_total = pd.concat([df_total, df_gen], ignore_index=True)
                else:
                    df_total = df_gen

                # Save merged CSV
                df_total.to_csv(predictions_csv_path, index=False)

                # Delete individual generator CSV
                os.remove(generator_csv_path)

        mean_acc = float(np.mean(accs))
        mean_ap = float(np.mean(aps))
        mean_auc = float(np.mean(aucs))
        mean_loss = float(np.mean(losss))
        
        # ================= OUTPUT PATHS PER MODEL =================
        output_dir = os.path.join(config.OUTPUT, f"epoch_{checkpoint_epoch}")
        os.makedirs(output_dir, exist_ok=True)
        metrics_csv_epoch = os.path.join(output_dir, "metrics.csv")
        # ================= SAVE METRICS =================

        if metrics_csv_epoch is not None:
            os.makedirs(os.path.dirname(metrics_csv_epoch), exist_ok=True)

            with open(metrics_csv_epoch, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["generator", "acc", "ap", "auc", "loss"])

                for g, m in log.items():
                    writer.writerow([
                        g,
                        m["acc"],
                        m["ap"],
                        m["auc"],
                        m["loss"],
                    ])

                writer.writerow([
                    "mean",
                    mean_acc,
                    mean_ap,
                    mean_auc,
                    mean_loss,
                ])

        # ================= SAVE RESULTS =================
        """
        if results_csv_epoch is not None:
            os.makedirs(os.path.dirname(results_csv_epoch), exist_ok=True)

            with open(results_csv_epoch, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["image_path", "generator", "probability", "label"],
                )
                writer.writeheader()
                writer.writerows(csv_rows)
        """
        logger.info(f"Saved metrics: {metrics_csv_epoch}")
        logger.info(f"Saved predictions: {predictions_csv_path}")


@cli.command()
@click.option("--method", type=str, multiple=True)
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-size", type=int,
              help="Batch size for a single GPU.")
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--model", "model_path",
              type=click.Path(exists=True, dir_okay=False),
              help="path to pre-trained model")
@click.option("--output", "output_dir", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--resize-to", type=int,
              help="When this argument is provided the testing images will be resized "
                   "so that their biggest dimension does not exceed this value.")
@click.option("--device", "device_name", type=str, default="cuda")
@click.option("--csv_delimiter", type=str, default=",")
@click.option("--results-csv-name", type=str, default=None,
              help="Filename of the CSV where the results will be written. Useful when "
                   "different runs on the the same model's checkpoint run in parallel for "
                   "different datasets, in order to not corrupt the CSV due to parallel write.")
def explain(
    method: list[str],
    cfg: Path,
    batch_size: Optional[int],
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    model_path: Path,
    output_dir: Path,
    tag: str,
    resize_to: Optional[int],
    device_name: str,
    csv_delimiter: str,
    results_csv_name: Optional[str]
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "output": str(output_dir),
        "tag": tag,
        "pretrained": str(model_path),
        "resize_to": resize_to
    })
    device: torch.device = torch.device(device_name)

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of Params: {n_parameters}")
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

    load_pretrained(config, model, logger)
    test_datasets_names, test_datasets, test_loaders = build_loader_test(config, logger)

    model.eval()

    output_dir: Path = output_dir / tag / "explainability" / Path(model_path).stem

    explainability.explain_model(
        model,
        output_dir,
        csv_delimiter,
        logger,
        test_loaders,
        test_datasets,
        test_datasets_names,
        device,
        results_csv_name=results_csv_name,
        expl_methods=tuple(method)
    )

    if log_writer is not None:
        log_writer.flush()


@cli.command(help="Benchmarks the runtime of a model across different image sizes.")
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--start-size", type=int, default=224,
              help="Start size for each dimension of the test images.")
@click.option("--increase-factor", type=float, default=2.0,
              help="The factor by which the size of each dimension of the image will "
                   "be multiplied.")
@click.option("--increase-steps", type=int, default=8,
              help="Number of times the image size will be increased.")
@click.option("--repeats", type=int, default=30,
              help="Number of images to be inferred for each size step.")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
@click.option("--device", type=str, default="cuda")
@torch.no_grad()
def runtime(
    cfg: Path,
    start_size: int,
    increase_factor: float,
    increase_steps: int,
    repeats: int,
    output: Path,
    tag: str,
    extra_options: tuple[str, str],
    device: str
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "output": str(output),
        "tag": tag,
        "opts": extra_options
    })

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    device: torch.device = torch.device(device)

    # Export current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())

    neptune_tags: list[str] = ["mfm", "runtime"]
    neptune_run = neptune.init_run(
        name=config.TAG,
        tags=neptune_tags
    )

    if log_writer is not None:
        log_writer.flush()
    if neptune_run is not None:
        neptune_run.sync()

    dim_size: int = start_size
    entries: list[dict[str, Any]] = []
    out_file: Path = Path(config.OUTPUT) / "runtime.csv"

    if config.MODEL.RESOLUTION_MODE == "arbitrary":
        logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        model.cuda()
        logger.info(str(model))
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of Params: {n_parameters}")
        if hasattr(model, "flops"):
            flops = model.flops()
            logger.info(f"Number of GFLOPs: {flops / 1e9}")

        for _ in tqdm(list(range(increase_steps)), desc=f"Computing runtime - Repeats: {repeats}"):
            # Generate a random image. Content of the image does not matter for runtime, only size.
            img: torch.Tensor = torch.rand((1, 3, dim_size, dim_size))
            img = img.to(device)

            # Warmup.
            for _ in range(5):
                model([img], config.MODEL.FEATURE_EXTRACTION_BATCH)

            # Forward pass the image the designated amount of times.
            torch.cuda.synchronize()
            start_time: float = time.time()
            for _ in range(repeats):
                model([img], config.MODEL.FEATURE_EXTRACTION_BATCH)
            torch.cuda.synchronize()
            stop_time: float = time.time()
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            # Generate a map between img size and per image runtime.
            img_size: float = (img.size(2) * img.size(3)) / 1_000_000
            elapsed: float = (stop_time - start_time) / repeats
            entries.append({
                "img_size": img_size,
                "runtime": elapsed,
                "memory": memory_used
            })
            neptune_run["runtime"].append(elapsed)
            data_utils.write_csv_file(entries, out_file, delimiter=",")

            dim_size = int(dim_size * increase_factor)
            torch.cuda.reset_max_memory_allocated()
    elif config.MODEL.RESOLUTION_MODE == "fixed":
        for _ in tqdm(list(range(increase_steps)), desc=f"Computing runtime - Repeats: {repeats}"):
            # Rebuild the model for each new resolution.
            config.defrost()
            config.DATA.IMG_SIZE = dim_size
            config.freeze()
            model = build_cls_model(config)
            model.cuda()

            # Generate a random image. Content of the image does not matter for runtime, only size.
            img: torch.Tensor = torch.rand((1, 3, dim_size, dim_size))
            img = img.to(device)

            # Warmup.
            for _ in range(5):
                model(img)
            torch.cuda.empty_cache()

            # Forward pass the image the designated amount of times.
            torch.cuda.synchronize()
            start_time: float = time.time()
            for _ in range(repeats):
                model(img)
            torch.cuda.synchronize()
            stop_time: float = time.time()
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            torch.cuda.empty_cache()

            # Generate a map between img size and per image runtime.
            img_size: float = (img.size(2) * img.size(3)) / 1_000_000
            elapsed: float = (stop_time - start_time) / repeats
            entries.append({
                "img_size": img_size,
                "runtime": elapsed,
                "memory": memory_used
            })
            neptune_run["runtime"].append(elapsed)
            data_utils.write_csv_file(entries, out_file, delimiter=",")

            dim_size = int(dim_size * increase_factor)
            torch.cuda.reset_max_memory_allocated()
    else:
        raise TypeError(f"Unsupported resolution mode: {config.MODEL.RESOLUTION_MODE}")


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--resize-to", type=int,
              help="When this argument is provided the testing images will be resized "
                   "so that their biggest dimension does not exceed this value.")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
def tsne(
    cfg: Path,
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    lmdb_path: Optional[Path],
    model: Path,
    output: Path,
    tag: str,
    resize_to: Optional[int],
    extra_options: tuple[str, str],
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": 1,  # Currently, required to be 1 for correctly distinguishing embeddings.
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "resize_to": resize_to,
        "opts": extra_options
    })
    from spai import tsne as tsne_utils

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())

    neptune_tags: list[str] = ["mfm", "tsne"]
    neptune_tags.extend([p.stem for p in test_csv])
    neptune_run = neptune.init_run(
        name=config.TAG,
        tags=neptune_tags
    )

    test_datasets_names, test_datasets, test_loaders = build_loader_test(config, logger)
    model_ckpt: pathlib.Path = find_pretrained_checkpoints(config)[0]

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    model.cuda()
    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of Params: {n_parameters}")
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

    checkpoint_epoch: int = load_pretrained(config, model, logger, checkpoint_path=model_ckpt)

    # Test the model.
    for test_data_loader, test_dataset, test_data_name in zip(test_loaders,
                                                              test_datasets,
                                                              test_datasets_names):
        tsne_utils.visualize_tsne(config, test_data_loader, test_data_name, model, neptune_run)

        if log_writer is not None:
            log_writer.flush()
        if neptune_run is not None:
            neptune_run.sync()


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
@click.option("--verbose", is_flag=True, help="Print verbose output.")
@click.option("--exclude-preprocessing", is_flag=True,
              help="When this flag is provided the exported encoder does not include the spectral "
                   "filtering and normalization preprocessing operations. Instead, it accepts "
                   "three inputs, requiring these operations to be previously performed.")
def export_onnx(
    cfg: Path,
    model: Path,
    output: Path,
    tag: str,
    extra_options: tuple[str, str],
    verbose: bool,
    exclude_preprocessing: bool
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "opts": extra_options
    })

    output: Path = Path(config.OUTPUT)
    output.mkdir(exist_ok=True, parents=True)

    global logger
    logger = create_logger(output_dir=output, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    config_export_path: Path = output / "config.json"
    with config_export_path.open("w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {config_export_path}")
    logger.info(config.dump())

    model_checkpoints: list[pathlib.Path] = find_pretrained_checkpoints(config)
    onnx_export_dir: Path = output / "onnx"
    onnx_export_dir.mkdir(exist_ok=True, parents=True)

    for i, model_ckpt in enumerate(model_checkpoints):
        if i == 0:
            logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        checkpoint_epoch: int = load_pretrained(config, model, logger,
                                                checkpoint_path=model_ckpt, verbose=i == 0)

        model.to("cpu")
        model.eval()

        patch_encoder: Path = onnx_export_dir / "patch_encoder.onnx"
        patch_aggregator: Path = onnx_export_dir / "patch_aggregator.onnx"
        model.export_onnx(patch_encoder, patch_aggregator,
                          include_fft_preprocessing=not exclude_preprocessing)


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-size", type=int, help="Batch size.")
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--split", type=str, default="test",
              help="The data split which will be tested. Actually, this value is expected to be "
                   "present in the `split` column of the provided csv files. Only samples "
                   "in the csv belonging to the provided split will be tested.")
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str, help="tag of experiment")
@click.option("--device", type=str, default="cpu")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
@click.option("--verbose", is_flag=True, help="Print verbose output.")
@click.option("--exclude-preprocessing", is_flag=True)
def validate_onnx(
    cfg: Path,
    batch_size: Optional[int],
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    split: str,
    lmdb_path: Optional[Path],
    model: Path,
    output: Path,
    tag: str,
    device: str,
    extra_options: tuple[str, str],
    verbose: bool,
    exclude_preprocessing: bool
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "opts": extra_options
    })

    output: Path = Path(config.OUTPUT)
    output.mkdir(exist_ok=True, parents=True)

    global logger
    logger = create_logger(output_dir=output, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    config_export_path: Path = output / "config.json"
    with config_export_path.open("w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {config_export_path}")
    logger.info(config.dump())

    test_datasets_names, test_datasets, test_loaders = build_loader_test(
        config, logger, split=split
    )

    model_checkpoints: list[pathlib.Path] = find_pretrained_checkpoints(config)
    onnx_export_dir: Path = output / "onnx"
    if not onnx_export_dir.exists():
        raise FileNotFoundError(f"No onnx model at {onnx_export_dir}. Use the export-onnx "
                                f"command first.")

    for i, model_ckpt in enumerate(model_checkpoints):
        if i == 0:
            logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        checkpoint_epoch: int = load_pretrained(config, model, logger,
                                                checkpoint_path=model_ckpt, verbose=i == 0)
        model.to(device)
        model.eval()

        patch_encoder: Path = onnx_export_dir / "patch_encoder.onnx"
        patch_aggregator: Path = onnx_export_dir / "patch_aggregator.onnx"

        compare_pytorch_onnx_models(
            model,
            patch_encoder,
            patch_aggregator,
            includes_preprocessing=not exclude_preprocessing,
            device=device
        )

        # # Test the model.
        # for test_data_loader, test_dataset, test_data_name in zip(test_loaders,
        #                                                           test_datasets,
        #                                                           test_datasets_names):
        #     predictions: Optional[dict[int, float]]
        #     acc, ap, auc, loss, predictions = validate(
        #         config, test_data_loader, model, criterion, neptune_run,
        #         return_predictions=True
        #     )
        #     logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch} | "
        #                 f"Images: {len(test_dataset)} | loss: {loss:.4f}")
        #     logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch}  | "
        #                 f"Images: {len(test_dataset)} | ACC: {acc:.3f}")
        #     logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch}  | "
        #                 f"Images: {len(test_dataset)} | AP: {ap:.3f}")
        #     logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch}  | "
        #                 f"Images: {len(test_dataset)} | AUC: {auc:.3f}")
        #     neptune_run[f"test/{test_data_name}/acc"].append(acc, step=checkpoint_epoch)
        #     neptune_run[f"test/{test_data_name}/ap"].append(ap, step=checkpoint_epoch)
        #     neptune_run[f"test/{test_data_name}/auc"].append(auc, step=checkpoint_epoch)
        #     neptune_run[f"test/{test_data_name}/loss"].append(loss, step=checkpoint_epoch)
        #
        #     if predictions is not None:
        #         column_name: str = f"{tag}_epoch_{checkpoint_epoch}"
        #         test_dataset.update_dataset_csv(
        #             column_name, predictions, export_dir=Path(config.OUTPUT)
        #         )


def train_model(
    config: yacs.config.CfgNode,
    model: nn.Module,
    model_without_ddp: nn.Module,
    data_loader_train: torch.utils.data.DataLoader,
    data_loader_val: torch.utils.data.DataLoader,
    data_loaders_test: list[torch.utils.data.DataLoader],
    dataset_val: spai.data.data_finetune.CSVDataset,
    datasets_test: list[spai.data.data_finetune.CSVDataset],
    datasets_test_names: list[str],
    criterion,
    optimizer,
    scaler,
    lr_scheduler,
    log_writer,
    save_all: bool = False
) -> None:
    logger.info("Start training")

    start_time: float = time.time()
    val_accuracy_per_epoch: list[float] = []
    val_ap_per_epoch: list[float] = []
    val_auc_per_epoch: list[float] = []
    val_loss_per_epoch: list[float] = []

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        epoch_start_time: float = time.time()
        
        train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            scaler,
            epoch,
            lr_scheduler,
            log_writer,
        )
        
        #neptune_run["train/last_epoch"] = epoch + 1
        #neptune_run["train/epochs_trained"] = epoch + 1 - config.TRAIN.START_EPOCH
        """
        # Validate the model.
        acc: float
        ap: float
        auc: float
        loss: float
        if config.TRAIN.MODE == "contrastive":
            acc, ap, auc, loss = validate_knn(config, data_loader_val, dataset_val, model)
        else:
            acc, ap, auc, loss = validate(config, data_loader_val, model, criterion, neptune_run)
        logger.info(f"Val | Epoch {epoch} | Images: {len(dataset_val)} | loss: {loss:.4f}")
        logger.info(f"Val | Epoch {epoch} | Images: {len(dataset_val)} | ACC: {acc:.3f}")
        logger.info(f"Val | Epoch {epoch} | Images: {len(dataset_val)} | AP: {ap:.3f}")
        logger.info(f"Val | Epoch {epoch} | Images: {len(dataset_val)} | AUC: {auc:.3f}")
        neptune_run["val/auc"].append(auc)
        neptune_run["val/ap"].append(ap)
        neptune_run["val/accuracy"].append(acc)
        neptune_run["val/loss"].append(loss)

        # Display the best epochs so far.
        val_accuracy_per_epoch.append(acc)
        val_ap_per_epoch.append(ap)
        val_auc_per_epoch.append(auc)
        val_loss_per_epoch.append(loss)
        logger.info(f"Val | Min loss: {min(val_loss_per_epoch):.4f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH + np.argmin(val_loss_per_epoch)}")
        logger.info(f"Val | Max ACC: {max(val_accuracy_per_epoch):.3f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH+np.argmax(val_accuracy_per_epoch)}")
        logger.info(f"Val | Max AP: {max(val_ap_per_epoch):.3f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH + np.argmax(val_ap_per_epoch)}")
        logger.info(f"Val | Max AUC: {max(val_auc_per_epoch):.3f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH + np.argmax(val_auc_per_epoch)}")

        # Save only the checkpoints that decrease validation loss.
        if len(val_loss_per_epoch) == 1 or loss < min(val_loss_per_epoch[:-1]) or save_all:
            save_checkpoint(config, epoch, model_without_ddp, max(val_accuracy_per_epoch),
                            optimizer, lr_scheduler, logger)
        """
        max_accuracy = None # max(val_accuracy_per_epoch)
        save_checkpoint(config, epoch, model_without_ddp, max_accuracy,
                            optimizer, lr_scheduler, logger,scaler)
        # Compute epoch time.
        epoch_time: float = time.time() - epoch_start_time
        logger.info(f"Epoch training time: {epoch_time:.3f}s")
        
        """
        accs, aps, aucs, losses = [], [], [], []
        # Test the model.
        for test_data_loader, test_dataset, test_data_name in zip(data_loaders_test,
                                                                  datasets_test,
                                                                  datasets_test_names):
            print(f"Testing on {test_data_name} dataset...")
            
            if config.TRAIN.MODE == "contrastive":
                acc, ap, auc, loss = validate_knn(config, test_data_loader, test_dataset, model)
            else:
                acc, ap, auc, loss = validate(config, test_data_loader, model,
                                              criterion, neptune_run)
            logger.info(f"Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| loss: {loss:.4f}")
            logger.info(f"Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| ACC: {acc:.3f}")
            logger.info(f"Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| AP: {ap:.3f}")
            logger.info(f"Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| AUC: {auc:.3f}")
            neptune_run[f"test/{test_data_name}/acc"].append(acc, step=epoch)
            neptune_run[f"test/{test_data_name}/ap"].append(ap, step=epoch)
            neptune_run[f"test/{test_data_name}/auc"].append(auc, step=epoch)
            neptune_run[f"test/{test_data_name}/loss"].append(loss if not np.isnan(loss) else -100., step=epoch)
            accs.append(acc)
            aps.append(ap)
            aucs.append(auc)
            losses.append(loss)
            
        neptune_run[f"test/mean/acc"].append(np.mean(accs), step=epoch)
        neptune_run[f"test/mean/ap"].append(np.mean(aps), step=epoch)
        neptune_run[f"test/mean/auc"].append(np.mean(aucs), step=epoch)
        neptune_run[f"test/mean/loss"].append(np.mean(losses), step=epoch)
        neptune_run["train/epoch_train_time"].append(epoch_time)

        if neptune_run is not None:
            neptune_run.sync()
        """
    # Compute total training time.
    total_time: float = time.time() - start_time
    total_time_str: str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Overall training time: {total_time_str}")
    #neptune_run["train/total_train_time"].append(total_time_str)


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    scaler,
    epoch,
    lr_scheduler,
    log_writer,
):
    use_amp = config.AMP_OPT_LEVEL != "O0"
    model.train()
    criterion.train()
    optimizer.zero_grad()
    
    logger.info(
        "Current learning rate for different parameter groups: "
        f"{[it['lr'] for it in optimizer.param_groups]}"
    )

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, batch in enumerate(data_loader):
        grad_norm = 0.0
        if isinstance(criterion, TripletMarginLoss):
            anchor, positive, negative = batch
            batch_size: int = anchor.size(0)
            anchor = anchor.cuda(non_blocking=True)
            positive = positive.cuda(non_blocking=True)
            negative = negative.cuda(non_blocking=True)
            with autocast(device_type="cuda", enabled=use_amp):
                anchor_outputs = model(anchor)
                positive_outputs = model(positive)
                negative_outputs = model(negative)
                loss = criterion(anchor_outputs, positive_outputs, negative_outputs)

        else:
            samples, targets, _ = batch
            batch_size: int = samples.size(0)
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            # Forward pass each augmented view of the batch separately in order to not
            # significantly increase memory requirements.
            with autocast(device_type="cuda", enabled=use_amp):
                outputs_views: list[torch.Tensor] = [
                    model(samples[:, i, :, :, :]) for i in range(samples.size(1))
                ]          
                outputs: torch.Tensor = torch.stack(outputs_views, dim=1)
                outputs = outputs if outputs.size(dim=1) > 1 else outputs.squeeze(dim=1)
                loss = criterion(outputs.squeeze(), targets)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
        scaler.scale(loss).backward()

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0: 
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD
                )
            else:
                scaler.unscale_(optimizer)
                grad_norm = get_grad_norm(model.parameters())
            # Optional: unscale before clipping
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), batch_size)
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[-1]["lr"]
        loss_value_reduce = loss.cpu().detach().numpy()
        grad_norm_cpu = (grad_norm.cpu().detach().numpy()
                         if isinstance(grad_norm, torch.Tensor) else grad_norm)

        if log_writer is not None and (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((idx / num_steps + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('grad_norm', grad_norm_cpu, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
                        
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(
    config,
    data_loader,
    model,
    criterion,
    neptune_run,
    verbose: bool = True,
    return_predictions: bool = False
):
    model.eval()
    criterion.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_metrics: metrics.Metrics = metrics.Metrics(metrics=("auc", "ap", "accuracy"))

    predicted_scores: dict[int, tuple[float, Optional[AttentionMask]]] = {}

    end = time.time()
    for idx, (images, target, dataset_idx) in enumerate(data_loader):
        if isinstance(images, list):
            # In case of arbitrary resolution models the batch is provided as a list of tensors.
            images = [img.cuda(non_blocking=True) for img in images]
            # Remove views dimension. Always 1 during inference.
            images = [img.squeeze(dim=1) for img in images]
        else:
            images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Compute output.
        if isinstance(images, list) and config.TEST.EXPORT_IMAGE_PATCHES:
            export_dirs: list[pathlib.Path] = [
                pathlib.Path(config.OUTPUT)/"images"/f"{dataset_idx.detach().cpu().tolist()[i]}"
                for i in range(len(dataset_idx))
            ]
            output, attention_masks = model(
                images, config.MODEL.FEATURE_EXTRACTION_BATCH, export_dirs
            )
        elif isinstance(images, list):
            output = model(images, config.MODEL.FEATURE_EXTRACTION_BATCH)
            attention_masks = [None] * len(images)
        else:
            if images.size(dim=1) > 1:
                predictions: list[torch.Tensor] = [
                    model(images[:, i]) for i in range(images.size(dim=1))
                ]
                predictions: torch.Tensor = torch.stack(predictions, dim=1)
                if config.TEST.VIEWS_REDUCTION_APPROACH == "max":
                    output: torch.Tensor = predictions.max(dim=1).values
                elif config.TEST.VIEWS_REDUCTION_APPROACH == "mean":
                    output: torch.Tensor = predictions.mean(dim=1)
                else:
                    raise TypeError(f"{config.TEST.VIEWS_REDUCTION_APPROACH} is not a "
                                    f"supported views reduction approach")
            else:
                images = images.squeeze(dim=1)  # Remove views dimension.
                output = model(images)
            attention_masks = [None] * images.size(0)

        loss = criterion(output.squeeze(dim=1), target)

        # Apply sigmoid to output.
        output = torch.sigmoid(output)

        # Update metrics.
        loss_meter.update(loss.item(), target.size(0))
        cls_metrics.update(output[:, 0].cpu(), target.cpu())

        # Keep predictions if requested.
        if return_predictions:
            batch_predictions: list[float] = output.squeeze(dim=1).detach().cpu().tolist()
            batch_dataset_idx: list[int] = dataset_idx.detach().cpu().tolist()
            predicted_scores.update({
                i: (p, m) for i, p, m in zip(batch_dataset_idx, batch_predictions, attention_masks)
            })

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 and verbose:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}] | '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | '
                f'Mem {memory_used:.0f}MB')

    metric_values: dict[str, np.ndarray] = cls_metrics.compute()
    auc: float = metric_values["auc"].item()
    ap: float = metric_values["ap"].item()
    acc: float = metric_values["accuracy"].item()

    if return_predictions:
        return acc, ap, auc, loss_meter.avg, predicted_scores
    else:
        return acc, ap, auc, loss_meter.avg


@torch.no_grad()
def validate_knn(config, data_loader, dataset, model) -> tuple[float, float, float, float]:
    model.eval()

    auc_metric = torchmetrics.classification.MulticlassAUROC(
        num_classes=dataset.get_classes_num()
    )
    ap_metric = torchmetrics.classification.MulticlassAveragePrecision(
        num_classes=dataset.get_classes_num()
    )
    acc_metric = torchmetrics.classification.MulticlassAccuracy(
        num_classes=dataset.get_classes_num()
    )

    start = time.time()

    # Compute the embeddings for all the test samples.
    embeddings: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for idx, (images, target, dataset_idx) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        targets.append(target)
        images = images.squeeze(dim=1)  # Remove views dimension. Always 1 during inference.
        # Compute output.
        output: torch.Tensor = model(images)  # B x D
        embeddings.append(output)
    embeddings: torch.Tensor = torch.cat(embeddings, dim=0)
    targets: torch.Tensor = torch.cat(targets, dim=0).long()

    # Compute the distances between all the embeddings.
    if config.TRAIN.LOSS == "triplet":
        pdist = nn.PairwiseDistance().cuda()
        max_dist: bool = False
    elif config.TRAIN.LOSS == "supcont":
        pdist = nn.CosineSimilarity().cuda()
        max_dist: bool = True
    else:
        raise RuntimeError(f"Unsupported loss for knn: {config.TRAIN.LOSS}")
    distances: torch.Tensor = torch.zeros((embeddings.shape[0], embeddings.shape[0])).cuda()
    for i in range(embeddings.shape[0]):
        distances[i, :] = pdist(embeddings[i, :], embeddings)
    # Remove distance with itself.
    if max_dist:
        distances.fill_diagonal_(.0)
    else:
        distances.fill_diagonal_(distances.max().item())

    # Compute top-10 neighbors for each embedding and use majority voting to predict class.
    nearest: torch.return_types.topk = torch.topk(distances, 10, dim=1, largest=max_dist)
    nearest_labels: torch.Tensor = torch.take(targets, nearest.indices)
    predicted_labels: torch.Tensor = torch.mode(nearest_labels, dim=1).values

    # Computer loss as the difference between intra and inter class distances.
    if max_dist:
        distances.fill_diagonal_(1.0)  # Set similarity with itself to 1.
    else:
        distances.fill_diagonal_(.0)  # Set distance with itself to 0.
    intra_class_dist: torch.Tensor = torch.zeros((1,)).cuda()
    inter_class_dist: torch.Tensor = torch.zeros((1,)).cuda()
    for i in range(dataset.get_classes_num()):
        # Compute average intra class distance.
        class_indices: torch.Tensor = targets == i
        class_indices = class_indices.unsqueeze(dim=1)
        distances_indices: torch.Tensor = class_indices * class_indices.T
        intra_class_dist += ((distances * distances_indices).sum()
                             / distances_indices.sum()
                             / dataset.get_classes_num())
        # Compute average inter class distance.
        distances_indices = class_indices * ~class_indices.T
        inter_class_dist += ((distances * distances_indices).sum()
                             / distances_indices.sum()
                             / dataset.get_classes_num())
    if max_dist:
        loss: float = (inter_class_dist.cpu().detach().item()
                       - intra_class_dist.cpu().detach().item()
                       + 1.0)
    else:
        loss: float = (intra_class_dist.cpu().detach().item()
                       - inter_class_dist.cpu().detach().item()
                       + config.TRAIN.TRIPLET_LOSS_MARGIN)

    # Update metrics.
    predicted_labels = torch.nn.functional.one_hot(
        predicted_labels.long(), num_classes=dataset.get_classes_num()
    ).float().cpu()
    targets = targets.cpu()
    auc_metric.update(predicted_labels, targets)
    ap_metric.update(predicted_labels, targets)
    acc_metric.update(predicted_labels, targets)

    # Measure elapsed time.
    eval_time: float = time.time() - start

    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f'Test: '
        f'Time {eval_time:.3f}s | '
        f'Loss (l2-dist) {loss:.4f}) | '
        f'Mem {memory_used:.0f}MB')

    auc: float = auc_metric.compute().item()
    ap: float = ap_metric.compute().item()
    acc: float = acc_metric.compute().item()

    return acc, ap, auc, loss


if __name__ == '__main__':
    cli()
