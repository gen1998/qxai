import os
import random

import cv2
import numpy as np
import pandas as pd
import torch

from . import BASE_PATCH_IMAGE_PATH


def data_load(
    ver: str = "split", bio_rate: float = 1.0, sur_rate: float = 1.0
) -> pd.DataFrame:
    df_input = pd.read_csv("./dataset/csv/input.csv")

    df_surgery = df_input[df_input.category == "surgery"]
    df_biopsy = df_input[df_input.category == "biopsy"]
    num_surgery = int(len(df_surgery) * sur_rate)
    num_biopsy = int(len(df_biopsy) * bio_rate)
    df_surgery = df_surgery.sample(num_surgery).reset_index(drop=True)
    df_biopsy = df_biopsy.sample(num_biopsy).reset_index(drop=True)
    df_input = pd.concat([df_surgery, df_biopsy])

    output = pd.DataFrame()

    for i, row in df_input.iterrows():
        one_label_df = pd.DataFrame()
        name = row["img_name"]
        count = row["count"]

        one_label_df["img_path"] = [
            os.path.join(BASE_PATCH_IMAGE_PATH, f"{i}_{name}") for i in range(count)
        ]
        one_label_df["img_name_f"] = [f"{i}_{name}" for i in range(count)]
        one_label_df["label"] = row["label"]
        if ver == "split":
            one_label_df["split"] = row["split"]
        elif ver == "fold":
            one_label_df["fold"] = row["fold"]
        one_label_df["surgery"] = row["surgery"]

        output = pd.concat([output, one_label_df])

    output = output.reset_index(drop=True)

    return output


def seed_everything(seed):
    "seed値を一括指定"
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_img(path):
    """
    pathからimageの配列を得る
    """
    im_bgr = cv2.imread(path)
    if im_bgr is None:
        print(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


def calc_black_whiteArea(bw_image):
    image_size = bw_image.size
    whitePixels = cv2.countNonZero(bw_image)
    whiteAreaRatio = whitePixels / image_size

    return whiteAreaRatio


def column_to_dict(df, key_name, value_name):
    output = df[[key_name, value_name]]
    output.index = output[key_name]
    output = output.drop([key_name], axis=1)
    output = output.to_dict()[value_name]
    return output
