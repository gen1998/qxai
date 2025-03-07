import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.dataset import set_kmeans_dataloader
from src.info import create_cluster_df, create_img_df
from src.model import (
    HepaClassifier,
    hierarchyclustering_train_infer,
    inference_embeddings,
)
from src.utils import column_to_dict, data_load


def calcurate_result(df, num_img):
    df_result = df.groupby("img_name").sum()
    df_result["result_"] = df_result["result"] * 64 / df_result["num"]
    df_result["label_"] = df_result["label"].apply(lambda x: 1 if x > 0 else 0)
    df_result = df_result[["result_", "label_"]]

    df_result = df_result.reset_index(drop=True)
    buff_0 = df_result[df_result.label_ == 0].result_.values
    buff_1 = df_result[df_result.label_ == 1].result_.values

    # validationの閾値を求める
    val_list = []
    for th in range(0, 64):
        val_list.append(
            (len(buff_0[buff_0 <= th]) + len(buff_1[buff_1 > th])) / num_img
        )

    th_max = np.argmax(val_list)

    return th_max, val_list, len(buff_0) + len(buff_1)


def kmeans_split(Config, device):
    if Config.log:
        logging.basicConfig(
            filename=f"./logs/log_{Config.project_name}.log",
            level=logging.DEBUG,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.debug("Kmeans Phase")

    print("-Kmeans Phase-")
    print()

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


def select_cluster_split(Config):
    df_img = create_img_df(Config.project_name)
    df_cluster = create_cluster_df(df_img)
    df_cluster_under10 = df_cluster[df_cluster.num < 10].reset_index(drop=True)
    df_cluster = df_cluster[df_cluster.num >= 10].reset_index(drop=True)

    if Config.log:
        logging.debug("Selection cluster Phase")
        logging.debug(f"Number of All clusters : {len(df_cluster)}")

    print("^Selection cluster Phase^")
    print(f"Number of All clusters : {len(df_cluster)}")

    num_img = len(df_img.img_name.unique())
    max_accuracy = 0.0
    index = 0
    up = 10

    # 10up
    while True:
        clusters = df_cluster.sort_values("correct_rate").cluster_name.values[index:]
        df = df_img[df_img.cluster_name.isin(clusters)].reset_index(drop=True)
        th_max, val_list, c_img = calcurate_result(df, num_img)

        if max_accuracy <= val_list[th_max]:
            max_accuracy = val_list[th_max]
        elif max_accuracy - 0.005 <= val_list[th_max]:
            index += up
            continue
        else:
            break

        if c_img < num_img:
            break

        index += up

    max_accuracy = 0.0
    index = index - 10
    end = index + 30
    up = 1
    max_accuracy = 0.0

    # 1up
    while True:
        clusters = df_cluster.sort_values("correct_rate").cluster_name.values[index:]
        df = df_img[df_img.cluster_name.isin(clusters)].reset_index(drop=True)

        th_max, val_list, c_img = calcurate_result(df, num_img)

        if max_accuracy <= val_list[th_max]:
            max_accuracy = val_list[th_max]
        else:
            break

        if c_img < num_img:
            break

        if index >= end:
            break

        index += up

    df_cluster["remove"] = 0
    df_cluster_under10["remove"] = 1
    df_cluster = df_cluster.sort_values("correct_rate").reset_index(drop=True)
    df_cluster.loc[: index - 2, "remove"] = 1
    df_cluster = pd.concat([df_cluster, df_cluster_under10])
    df_cluster = df_cluster[["cluster_name", "remove"]]

    df_cluster.to_csv(f"./result/clusters/{Config.project_name}_selected_clusters.csv")

    if Config.log:
        logging.debug(
            f"Number of Remain clusters : {len(df_cluster[df_cluster.remove == 0])}"
        )
        logging.debug(
            f"Selecting Cluster filename : {Config.project_name}_selected_clusters.csv"
        )

    print(f"Number of Remain clusters : {len(df_cluster[df_cluster.remove == 0])}")
    print()
