import sys

sys.path.append("pytorch-image-models/")

import lightgbm as lgb
import numpy as np
import pandas as pd
import timm
import torch
from sklearn import metrics
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    from cuml.cluster import KMeans as KMeans
except:
    pass


class HepaClassifier(nn.Module):
    """Classification Model"""

    def __init__(self, model_arch: str, n_class: int = 2, pretrained: bool = True):
        """
        Args:
            model_arch (str): model architecture name
            n_class (int): number of classification
            model_shape (str): Due to the model's structure, the connection layers change.
            pretrained (bool, optional): pretrained model or not. Defaults to True.
        """
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)

        if hasattr(self.backbone, "classifier"):
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, n_class)
        elif hasattr(self.backbone, "head"):
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, n_class)
        elif hasattr(self.backbone, "fc"):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class EarlyStopping:
    """Early Stopping Class"""

    def __init__(self, patience: int):
        """
        Args:
            patience (int): How many epochs should be tolerated without seeing a value update
        """
        self.max_val_monitor = 1000
        self.min_val_monitor = -1000
        self.val_epoch = -1
        self.stop_count = 0
        self.patience = patience
        self.min_delta = 0

    # mode = "min" or "max"(val_loss, val_accuracy)
    def update(self, monitor, epoch, mode):
        if mode == "max":
            if monitor > self.min_val_monitor:
                self.min_val_monitor = monitor
                self.val_epoch = epoch
                self.stop_count = 0
            else:
                self.stop_count += 1
        else:
            if monitor < self.max_val_monitor:
                self.max_val_monitor = monitor
                self.val_epoch = epoch
                self.stop_count = 0
            else:
                self.stop_count += 1

        if self.stop_count >= self.patience:
            return -1
        else:
            return 0


def train_one_epoch(
    epoch,
    model,
    loss_fn,
    optimizer,
    train_loader,
    device,
    verbose_step=1,
    scheduler=None,
    schd_batch_update=False,
):
    """train one epoch"""
    model.train()
    scaler = GradScaler()

    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        with autocast():
            image_preds = model(imgs)
            loss = loss_fn(image_preds, image_labels)

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * 0.99 + loss.item() * 0.01

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None and schd_batch_update:
                scheduler.step()

            if ((step + 1) % verbose_step == 0) or ((step + 1) == len(train_loader)):
                description = f"epoch {epoch} loss: {running_loss:.4f}"
                pbar.set_description(description)

    print("train: " + description)
    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(
    epoch,
    model,
    loss_fn,
    val_loader,
    device,
    verbose_step=1,
    scheduler=None,
    schd_loss_update=False,
):
    """valid one epoch"""
    model.eval()

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        image_preds = model(imgs)

        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % verbose_step == 0) or ((step + 1) == len(val_loader)):
            description = f"epoch {epoch} loss: {loss_sum / sample_num:.4f}"
            pbar.set_description(description)

    print("valid " + description)
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print(
        "validation multi-class accuracy = {:.4f}".format(
            (image_preds_all == image_targets_all).mean()
        )
    )

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()

    monitor = {}
    monitor["val_loss"] = loss_sum / sample_num
    monitor["val_accuracy"] = (image_preds_all == image_targets_all).mean()
    return monitor


def inference_one_epoch(model, data_loader, device):
    """inference one epochs"""
    model.eval()

    image_preds_all = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs, _) in pbar:
        imgs = imgs.to(device).float()

        image_preds = model(imgs)  # output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


def inference_embeddings(model, data_loader, device):
    """extract embeddings"""
    model.eval()
    image_preds_all = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    for step, (imgs, _) in pbar:
        imgs = imgs.to(device).float()

        model.model.global_pool.register_forward_hook(get_activation("act2"))
        image_preds = model(imgs)
        image_preds_all += [torch.softmax(activation["act2"], 1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    print(image_preds_all.shape)
    return image_preds_all


def get_image_predictions_kmeans(
    df_tst, df_val, test_embeddings, val_embeddings, n_clusters, index
):
    """train and predict cluster
    val : train
    test : predict
    """
    if index == 0:
        model = KMeans(n_clusters=n_clusters, max_iter=300, init="scalable-k-means++")
        model.fit(val_embeddings)
        df_val[f"labels_{index}"] = model.labels_
        df_tst[f"labels_{index}"] = model.predict(test_embeddings)
    else:
        df_val[f"labels_{index}"] = -100
        df_tst[f"labels_{index}"] = -100
        indexes = df_val[f"labels_{index - 1}"].unique()
        for i in indexes:
            buff = df_val[df_val[f"labels_{index - 1}"] == i]
            if len(buff) < 10 or i == -100:
                continue
            p = len(buff[buff["label"] == 0]) / len(buff)
            if p > 0.95 or p < 0.05:
                continue
            attention_index_val = np.where(df_val[f"labels_{index - 1}"] == i)
            attention_index_tst = np.where(df_tst[f"labels_{index - 1}"] == i)
            embeddings_val = val_embeddings[attention_index_val]
            embeddings_tst = test_embeddings[attention_index_tst]
            if len(embeddings_val) < 1:
                continue
            model = KMeans(
                n_clusters=n_clusters, max_iter=300, init="scalable-k-means++"
            )
            model.fit(embeddings_val)
            df_val.loc[attention_index_val[0], f"labels_{index}"] = (
                model.labels_ + 2 * i
            )
            if len(embeddings_tst) < 1:
                continue
            df_tst.loc[attention_index_tst[0], f"labels_{index}"] = (
                model.predict(embeddings_tst) + 2 * i
            )

    return df_val, df_tst


def step_one_clustering(
    df_tst, df_val, val_embeddings, test_embeddings, index, hie, n_clusters=2
):
    attention_index_val = np.where(df_val[f"labels_{index - 1}"] == hie)
    attention_index_tst = np.where(df_tst[f"labels_{index - 1}"] == hie)
    embeddings_val = val_embeddings[attention_index_val]
    embeddings_tst = test_embeddings[attention_index_tst]

    model = KMeans(n_clusters=n_clusters, max_iter=300, init="scalable-k-means++")
    model.fit(embeddings_val)
    df_val.loc[attention_index_val[0], f"labels_{index}"] = model.labels_ + 2 * hie

    if len(embeddings_tst) > 0:
        df_tst.loc[attention_index_tst[0], f"labels_{index}"] = (
            model.predict(embeddings_tst) + 2 * hie
        )

    return df_tst, df_val


def check_divided_lightgbm(buff, features):
    img_names = buff.img_name_f.values

    y_train = features.img_name.isin(img_names).astype(int).values
    if "split" in features.columns:
        train_feature = features.drop(
            ["label", "img_name", "origin_name", "split"], axis=1
        )
    else:
        train_feature = features.drop(
            ["label", "img_name", "origin_name", "fold"], axis=1
        )

    x_train = train_feature.copy()
    lgbtrain = lgb.Dataset(x_train, label=y_train)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
    }

    gbm = lgb.train(param, lgbtrain)
    preds = gbm.predict(x_train)
    pred_labels = np.rint(preds)

    f1 = metrics.f1_score(y_train, pred_labels)

    if f1 > 0.9:
        return 1
    else:
        return 0


def hierarchyclustering_train_infer(
    df_tst, df_val, features, test_embeddings, val_embeddings, n_clusters, index
):
    if index == 0:
        model = KMeans(n_clusters=n_clusters, max_iter=300, init="scalable-k-means++")
        model.fit(val_embeddings)
        df_val[f"labels_{index}"] = model.labels_
        df_tst[f"labels_{index}"] = model.predict(test_embeddings)
    else:
        df_val[f"labels_{index}"] = -100
        df_tst[f"labels_{index}"] = -100
        hierarchy = df_val[f"labels_{index - 1}"].unique()
        for hie in hierarchy:
            # 前の分離で分けられていないクラスタはスキップ
            if hie == -100:
                continue

            buff = df_val[df_val[f"labels_{index - 1}"] == hie]
            # 所属する枚数が10枚以下のクラスタはスキップ
            if len(buff) < 10:
                continue

            p = len(buff[buff["label"] == 0]) / len(buff)
            # どちらかに偏っている場合はスキップ
            if p > 0.95 or p < 0.05:
                # 150枚以下は問答無用にスキップ
                if len(buff) < 150:
                    continue
                check_lgb = check_divided_lightgbm(buff, features)
                # 分けられるクラスタはスキップ
                if check_lgb == 1:
                    continue

            # 二つに分けるクラスタリングをするだけ
            df_tst, df_val = step_one_clustering(
                df_tst, df_val, val_embeddings, test_embeddings, index, hie
            )

    return df_val, df_tst


def check_divided_lightgbm_importance(features, cluster_name, result):
    SELECT_NUM_FEATURES = [14, 7, 6, 5, 4, 3, 2, 1]
    y_train = (features.cluster_name == cluster_name).astype(int).values
    train_feature = features.drop(
        ["label", "img_name", "origin_name", "split", "cluster_name"], axis=1
    )

    index = 0

    while True:
        x_train = train_feature.copy()
        lgbtrain = lgb.Dataset(x_train, label=y_train)

        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
        }

        gbm = lgb.train(param, lgbtrain)
        preds = gbm.predict(x_train)
        pred_labels = np.rint(preds)

        f1 = metrics.f1_score(y_train, pred_labels, zero_division=0)

        if f1 < 0.9:
            break
        else:
            importance = pd.Series(
                gbm.feature_importance(importance_type="gain"), index=x_train.columns
            ).to_frame(name="重要度")
            importance["重要度"] /= importance["重要度"].sum() * (1 / 100)
            columns_buff = train_feature.columns.tolist()

            train_feature = train_feature[
                importance.sort_values("重要度", ascending=False).index[
                    : SELECT_NUM_FEATURES[index + 1]
                ]
            ]
            index += 1

    if index == 0:
        result[cluster_name] = [f1, -1]
    else:
        result[cluster_name] = [
            f1,
            len(columns_buff),
            columns_buff,
            importance["重要度"].values.tolist(),
        ]

    return result
