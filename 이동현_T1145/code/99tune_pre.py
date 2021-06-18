"""Tune Model.

- Author: Junghoon Kim, Jongsun Shin
- Contact: placidus36@gmail.com, shinn1897@makinarocks.ai
"""
import argparse
import copy
import optuna
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import yaml

from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info
from src.utils.common import read_yaml
from src.utils.macs import calc_macs
from src.trainer import TorchTrainer
from typing import Any, Dict, List, Tuple, Union
from train import train

MODEL_CONFIG = read_yaml(cfg="configs/model/mobilenetv3.yaml")
DATA_CONFIG = read_yaml(cfg="configs/data/taco.yaml")

def search_hyperparam(
    trial: optuna.trial.Trial,
) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    epochs = trial.suggest_int("epochs", low=20, high=50, step=5)
    img_size = trial.suggest_int("img_size", low=16, high=96, step=2)
    n_select = trial.suggest_int("n_select", low=0, high=5, step=1)
    batch_size = trial.suggest_int("batch_size", low=8, high=128)
    val_ratio = trial.suggest_float('val_ratio', low=0.1, high=0.2, log=True)
    init_lr = trial.suggest_float('init_lr', low=0.1, high=0.5, log=True)
    # init_lr = 1.5
    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size,
        "VAL_RATIO": val_ratio,
        "INIT_LR": init_lr
    }


def objective(
        trial: optuna.trial.Trial, 
        device,
        model_config,
        data_config
    ) -> Tuple[float, int, float]:
    """Optuna objective.

    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    # model_config = read_yaml(cfg=model_config)
    model_config = {}
    data_config = read_yaml(cfg=data_config)

    # hyperparams: EPOCHS, IMG_SIZE, n_select, BATCH_SIZE
    hyperparams = search_hyperparam(trial)

    model_config["input_size"] = [hyperparams["IMG_SIZE"], hyperparams["IMG_SIZE"]]
    # model_config["backbone"] = search_model(trial)
    # [추가]
    model_config["model_name"] = trial.suggest_categorical("model_name", [
            "mnasnet0_5",
            "shufflenet_v2_x0_5",
            # "squeezenet1_0",
        ]   
    )
    model_config['pretrained'] = trial.suggest_categorical('pretrained', [True, False])
    
    data_config["AUG_TRAIN_PARAMS"]["n_select"] = hyperparams["n_select"]
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["EPOCHS"] = hyperparams["EPOCHS"]
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]
    data_config["INIT_LR"] = hyperparams["INIT_LR"]
    data_config["VAL_RATIO"] = hyperparams["VAL_RATIO"]

    # Calculate macs
    min_macs = 1000000
    max_macs = 3200000
    model_instance = Model(model_config, verbose=False)
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))

    log_dir = os.path.join("exp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{int(macs)}")
    if min_macs <= macs <= max_macs:
        os.makedirs(log_dir, exist_ok=True)
        parmas_dir = os.path.join(log_dir, 'trial_params.yml')
        
        with open(parmas_dir, 'w') as f:
            yaml.dump(trial.params, f, default_flow_style=False)
    else:
        raise optuna.exceptions.TrialPruned()
    
    # model_config, data_config
    _, test_f1, _ = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )

    return test_f1, macs


def tune(
        gpu_id: int, 
        model_config: str,
        data_config: str,
        storage: Union[str, None] = None, 
        study_name: str = "pstage_automl",
    ):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")

    sampler = optuna.samplers.MOTPESampler(n_startup_trials=20)
    
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage=rdb_storage,
        study_name=study_name,
        sampler=sampler,
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, device, model_config, data_config), n_trials=2000)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", 
                        default="postgresql://optuna:t114599con@127.0.0.1:6006/pstage4", 
                        type=str, 
                        help="RDB Storage URL for optuna.")
    parser.add_argument("--study-name", default="pstage_automl", type=str, help="Optuna study name.")
    parser.add_argument("--model-config", 
                        default="configs/model/mobilenetv3.yaml", 
                        type=str, 
                        help="test set for model")
    parser.add_argument("--data-config", 
                        default="configs/data/taco.yaml", 
                        type=str, 
                        help="test set for data")

    args = parser.parse_args()
    tune(
        args.gpu, 
        model_config=args.model_config,
        data_config=args.data_config,
        storage=None if args.storage == "" else args.storage, 
        study_name=args.study_name
    )