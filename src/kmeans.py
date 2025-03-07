import logging

import pandas as pd
import torch
from tqdm import tqdm

from src.dataset import set_kmeans_dataloader
from src.model import (
    HepaClassifier,
    hierarchyclustering_train_infer,
    inference_embeddings,
)
from src.utils import column_to_dict, data_load


def kmeans_split(Config, device):
    if Config.log:
        logging.basicConfig(
            filename=f"./logs/log_{Config.project_name}.log",
            level=logging.DEBUG,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.debug("Kmeans Phase of Clustering")

    # train用 df の作成
    test_df = data_load(
        ver="split",
        bio_rate=Config.bio_rate,
        sur_rate=Config.sur_rate,
    )

    # infer
    tst_loader, test_df, val_loader, val_df = set_kmeans_dataloader(
        df=test_df,
        input_shape=Config.img_size,
        valid_bs=Config.valid_bs,
    )

    # features
    _input = pd.read_csv("./dataset/csv/input.csv")
    _input = _input[["img_name", "label", "split"]].reset_index(drop=True)
    cd_split = column_to_dict(_input, "img_name", "split")
    cd_label = column_to_dict(_input, "img_name", "label")
    del _input

    features = pd.read_csv("./result/features/percentile_features.csv")
    features.insert(
        1, "origin_name", features.img_name.apply(lambda x: x[x.find("_") + 1 :])
    )
    features.insert(2, "split", features.origin_name.apply(lambda x: cd_split[x]))
    features.insert(3, "label", features.origin_name.apply(lambda x: cd_label[x]))
    features = features[features.split == 1].reset_index(drop=True)

    model = HepaClassifier(
        model_arch=Config.model_arch,
        pretrained=Config.pretrained,
    ).to(device)
    model.load_state_dict(torch.load(f"save/{Config.project_name}/split_weight"))

    with torch.no_grad():
        tst_embeddings = inference_embeddings(model, tst_loader, device)
        val_embeddings = inference_embeddings(model, val_loader, device)

    divid_index = 60

    for index in tqdm(range(divid_index)):
        val_df, test_df = hierarchyclustering_train_infer(
            test_df, val_df, features, tst_embeddings, val_embeddings, index
        )
        test_df = test_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    test_df["ver"] = "test"
    val_df["ver"] = "val"
    output = pd.concat([test_df, val_df])
    output = output.reset_index(drop=True)

    output.to_csv(f"./result/predicitons/{Config.project_name}_kmeans.csv", index=False)
    if Config.log:
        logging.debug(f"Kmeans Result filename : {Config.project_name}_kmeans.csv")
