import json
import logging
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import BASE_HOVERNET_JSON_PATH, BASE_ROI_IMAGE_PATH


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def calcurate_feature(contours, otsu_img):
    area = []
    perimeter = []
    roundness = []

    for cnt in contours:
        a = cv2.contourArea(cnt)
        p = cv2.arcLength(cnt, True)
        area.append(a)
        perimeter.append(p)
        roundness.append(4 * np.pi * a / (p * p))

    nuclear = len(contours)
    if np.sum(otsu_img == 0) == 0:
        cytoplasm = 0
    else:
        cytoplasm = np.sum(otsu_img == 0)

    return area, perimeter, roundness, nuclear, cytoplasm


def calcurate_grayscale(contours, gray, x, y):
    chromatin = []
    contours = contours.copy()

    for cnt in contours:
        mask = np.zeros_like(gray)
        cnt[:, 0] = cnt[:, 0] - x
        cnt[:, 1] = cnt[:, 1] - y
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        pixels = gray[mask == 255]
        mean_color = np.mean(pixels)
        chromatin.append(round(mean_color, 2))

    return chromatin


def create_nuclear_features():
    biopsy_place = pd.read_csv("./dataset/csv/biopsy_place_list.csv", index_col=0)
    name_list = os.listdir(BASE_ROI_IMAGE_PATH)
    name_list = sorted(name_list)

    margin = 1
    size = 256
    output = {}

    for img_name in tqdm(name_list):
        img_name = img_name[:-4]
        with open(os.path.join(BASE_HOVERNET_JSON_PATH, f"{img_name}.json"), "r") as f:
            json_data = json.load(f)

        category = "s"
        index = 0

        if "N" not in img_name:
            img = cv2.imread(os.path.join(BASE_ROI_IMAGE_PATH, f"{img_name}.tif"))
            buff_count = biopsy_place.loc[img_name + ".tif", :]
            category = "b"
        else:
            img = cv2.imread(os.path.join(BASE_ROI_IMAGE_PATH, f"{img_name}.bmp"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        nuc_info = json_data["nuc"]
        contour_list = []
        for inst in nuc_info:
            inst_info = nuc_info[inst]
            contour_list.append(np.array(inst_info["contour"]))

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, otsu_thresholded = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        for y in range(8):
            for x in range(8):
                if buff_count[x + y * 8] == 1.0 or category == "s":
                    gray = img_gray[
                        y * size : (y + 1) * size, x * size : (x + 1) * size
                    ].copy()
                    x_ = x * 256 + margin
                    y_ = y * 256 + margin
                    contours_in_region = [
                        cnt
                        for cnt in contour_list
                        if cv2.boundingRect(cnt)[0] >= x_
                        and cv2.boundingRect(cnt)[1] >= y_
                        and cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2]
                        <= x_ + size
                        and cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3]
                        <= y_ + size
                    ]
                    otsu_img = otsu_thresholded[
                        y * size : (y + 1) * size, x * size : (x + 1) * size
                    ]
                    area, perimeter, roundness, nuclear, cytoplasm = calcurate_feature(
                        contours_in_region, otsu_img
                    )
                    chromatin = calcurate_grayscale(
                        contours_in_region, gray, x * size, y * size
                    )

                    output[f"{index}_{img_name}"] = {
                        "area": area,
                        "perimeter": perimeter,
                        "roundness": roundness,
                        "nuclear": nuclear,
                        "cytoplasm": cytoplasm,
                        "chromatin": chromatin,
                    }

                    index += 1

    with open("./result/features/nuclear_features.json", "w") as f:
        json.dump(output, f, cls=NpEncoder)


def create_percentile_features(area_th: int = 200, chr_th: int = 65):
    with open("./result/features/nuclear_features.json", "r") as f:
        json_data = json.load(f)

    output = []
    percentile_columns = ["area", "perimeter", "roundness", "chromatin"]
    percentiles = [25, 50, 75]

    for _, value in tqdm(json_data.items()):
        a = np.array(value["area"])
        c = np.array(value["chromatin"])
        indices = np.where((a >= area_th) | (c >= chr_th))[0]
        if value["nuclear"] > 0 and len(indices) > 0:
            buff = []
            for cl in percentile_columns:
                p = np.array(value[cl])[indices]
                if cl == "area":
                    area = np.sum(p)
                for per in percentiles:
                    buff.append(np.percentile(p, per))

            buff.append(value["nuclear"])
            buff.append(area / value["cytoplasm"])
        else:
            buff = [0 for _ in range(3 * 3 + 2)]

        output.append(buff)

    df_columns = [f"{cl}_{per}" for cl in percentile_columns for per in percentiles]
    df_columns.extend(["N_density", "N_C"])
    df = pd.DataFrame(output, columns=df_columns)
    df.insert(0, "img_name", list(json_data.keys()))
    df.img_name = df.img_name.apply(lambda x: x + ".bmp" if "N" in x else x + ".tif")

    df.to_csv("./result/features/percentile_features.csv", index=False)


def create_features(Config):
    if Config.log:
        logging.basicConfig(
            filename=f"./log/{Config.project_name}.log",
            level=logging.DEBUG,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.debug("feature generation phase")

    print("feature generation phase")
    print()

    create_nuclear_features()
    create_percentile_features()

    if Config.log:
        logging.debug("finish feature generation")
