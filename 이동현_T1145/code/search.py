import argparse
from datetime import datetime
import os
import yaml
from typing import Any, Dict, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import check_runtime, model_info

import optuna
from optuna.trial import TrialState


def search_model(
    trial: optuna.trial.Trial,
    model_config: Dict[str, Any]
) -> List[Any]:
    
    pass


def search_hyperparams(trial: optuna.trial.Trial) -> Dict[str, Any]:
    pass


def train_model(
    model: nn.Module,
    hyperparams: Dict[str, Any],
) -> nn.Module:
    pass


def evaluate(model: nn.Module) -> float:
    pass


def objective(
    trial: optuna.trial.Trial,
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
) -> float:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = search_model(trial)
    hyperparams = search_hyperparams(trial)
    model = train_model(Model(architecture, verbose=True), hyperparams)
    score = evaluate(model)
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--model",
                        default="configs/optuna/mobilenetv3_config.yaml",
                        type=str,
                        help="model config")
    parser.add_argument("--data",
                        default="configs/optuna/data_config.yaml",
                        type=str,
                        help="data config")

    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("exp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_config, data_config),
                   n_trials=200)
    print(study.best_trials)