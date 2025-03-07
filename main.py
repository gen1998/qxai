import argparse

import torch

from src.clustering import kmeans_split, select_cluster_split
from src.evaluate import evaluate_split, infer_grade
from src.feature import create_features
from src.train import run_fold, run_split
from src.utils import seed_everything


# argparse function
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log", action="store_true")

    # action setting
    parser.add_argument("--create_feature", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--kmeans", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    # data setting
    parser.add_argument("--project_name", default="qxai_split", type=str)
    parser.add_argument("--project_type", default="split", type=str)
    parser.add_argument("--bio_rate", default=1.0, type=float)
    parser.add_argument("--sur_rate", default=1.0, type=float)
    parser.add_argument("--img_size", default=256, type=int)

    # training model
    parser.add_argument("--model_arch", default="tf_efficientnet_b3", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--bs_size", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=3e-6, type=float)
    parser.add_argument("--er_patience", default=2, type=int)
    parser.add_argument("--er_mode", default="min", type=str)
    parser.add_argument("--er_monitor", default="val_loss", type=str)

    # 引数を解析
    return parser.parse_args()


def main():
    # config, global value setting
    Config = parse_arguments()
    print(Config)

    seed_everything(Config.seed)
    device = torch.device("cuda:0")

    if Config.create_feature:
        create_features(Config=Config)

    if Config.train:
        if Config.project_type == "fold":
            run_fold(Config=Config, device=device)
        elif Config.project_type == "split":
            run_split(Config=Config, device=device)

    if Config.kmeans:
        if Config.project_type == "fold":
            run_fold(Config=Config, device=device)
        elif Config.project_type == "split":
            kmeans_split(Config=Config, device=device)
            select_cluster_split(Config=Config)

    if Config.evaluate:
        if Config.project_type == "fold":
            run_fold(Config=Config, device=device)
        elif Config.project_type == "split":
            infer_grade(Config=Config)
            evaluate_split(Config=Config)


if __name__ == "__main__":
    main()
