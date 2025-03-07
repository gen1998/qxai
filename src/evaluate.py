import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

from src.info import create_cluster_df, create_img_df
from src.utils import column_to_dict


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
    return tn / (tn + fp)


def apply_infer_grade(row):
    if row.remove == 1:
        return "N_GRADE_R"
    else:
        return row.infer_grade


def apply_infer_grade_rate(row):
    remain = 64 - row.infer_grade_remain_N_GRADE_R

    if row.infer_grade_remain_N_GRADE / remain > 0.5:
        return "N_GRADE"

    compare = row["grade_G4":"grade_G1"]
    compare = compare.astype(int)
    output = compare.idxmax()
    output = output[output.rfind("_") + 1 :]

    if output in ["G1", "G2", "G3", "G4"]:
        return output


GRADE_TEXT_DICT = {
    1.0: "G1",
    1.5: "G1-G2",
    2.0: "G2",
    2.5: "G2-G3",
    3.0: "G3",
    3.5: "G3-G4",
    4.0: "G4",
    -1.0: "No Label",
}
TXT_GRADE_DICT = {
    "g_1": "G1",
    "g_2": "G2",
    "g_3": "G3",
    "g_4": "G4",
    "g_12": "G1-G2",
    "g_23": "G2-G3",
    "g_34": "G3-G4",
}

GRADE_LIST = ["G1", "G1-G2", "G2", "G2-G3", "G3", "G3-G4", "G4"]


def infer_grade(Config):
    # logging
    if Config.log:
        logging.basicConfig(
            filename=f"./logs/log_{Config.project_name}.log",
            level=logging.DEBUG,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.debug("Inference grade Phase")

    print("-Inference grade Phase-")

    # Grade infer Section
    df_img = create_img_df(Config.project_name)
    df_cluster = create_cluster_df(df_img)

    select_cluster = pd.read_csv(
        f"./result/clusters/{Config.project_name}_selected_clusters.csv"
    )
    select_cluster = column_to_dict(select_cluster, "cluster_name", "remove")
    df_cluster["remove"] = df_cluster.cluster_name.apply(lambda x: select_cluster[x])
    df_img["remove"] = df_img.cluster_name.apply(lambda x: select_cluster[x])
    cluster_name_list = df_cluster.cluster_name.values

    df_cluster = df_cluster[
        ((df_cluster.surgery > 0) & (df_cluster.label == 1) & (df_cluster.num > 10))
    ]
    df_cluster = (
        df_cluster.sort_values("g_4_p_per", ascending=False)
        .sort_values("g_3_p_per", ascending=False)
        .sort_values("g_2_p_per", ascending=False)
        .sort_values("g_1_p_per", ascending=False)
    )
    df_cluster = df_cluster.reset_index(drop=1)
    cluster_grade_dict_ = {}

    K = 0.75
    L = 0.74

    for _, row in df_cluster.iterrows():
        row_ = row["g_1_p_st_per":"g_4_p_st_per"]
        row_ = row_.astype(float)

        if row_.max() > K:
            txt = row_.idxmax()
            txt = txt[: txt.find("p") - 1]

            cluster_grade_dict_[row.cluster_name] = TXT_GRADE_DICT[txt]
            continue

        row_ = row["g_12_p_st_per":"g_34_p_st_per"]
        row_ = row_.astype(float)

        if row_.max() > L:
            txt = row_.idxmax()
            txt = txt[: txt.find("p") - 1]

            cluster_grade_dict_[row.cluster_name] = TXT_GRADE_DICT[txt]
            continue

    print(f"Number of grade label clusters : {len(cluster_grade_dict_)}")

    cluster_grade_dict = {}
    for c in cluster_name_list:
        cluster_grade_dict[c] = "N_GRADE"
    for c, v in cluster_grade_dict_.items():
        cluster_grade_dict[c] = v

    cluster_grade_dict = pd.DataFrame(
        cluster_grade_dict.values(), index=cluster_grade_dict.keys(), columns=["Grade"]
    )
    cluster_grade_dict = cluster_grade_dict.reset_index()
    cluster_grade_dict.columns = ["cluster_name", "Grade"]
    cluster_grade_dict.to_csv(
        f"./result/clusters/{Config.project_name}_grade_clusters.csv"
    )

    if Config.log:
        logging.debug(f"Number of grade label clusters : {len(cluster_grade_dict_)}")
        logging.debug(
            f"grade clusters filename : {Config.project_name}_grade_clusters.csv"
        )


def evaluate_split(Config):
    # Result showing Section
    if Config.log:
        logging.debug("-" * 100)
        logging.debug("Result Phase")

    print("-Result Phase-")

    ## Calculate the threshold using validation result
    df_img = create_img_df(Config.folder_name)
    df_cluster = create_cluster_df(df_img)

    # contributed cluster
    select_cluster = pd.read_csv(
        f"./result/clusters/{Config.project_name}_selected_clusters.csv"
    )
    select_cluster = column_to_dict(select_cluster, "cluster_name", "remove")
    df_cluster["remove"] = df_cluster.cluster_name.apply(lambda x: select_cluster[x])
    df_img["remove"] = df_img.cluster_name.apply(lambda x: select_cluster[x])

    # grade infer
    grade_cluster = pd.read_csv(
        f"./result/clusters/{Config.folder_name}_grade_clusters.csv"
    )
    grade_cluster = column_to_dict(grade_cluster, "cluster_name", "Grade")
    df_cluster["infer_grade"] = df_cluster.cluster_name.apply(
        lambda x: grade_cluster[x]
    )
    df_img["infer_grade"] = df_img.cluster_name.apply(lambda x: grade_cluster[x])
    df_img["place"] = df_img.img_name_f.apply(lambda x: int(x[: x.find("_")]))

    df_img_ = df_img[((df_img.label == 1) & (df_img.surgery == 1))]
    df_img_["infer_grade_remain"] = df_img_.apply(apply_infer_grade, axis=1)
    df_roi = pd.get_dummies(df_img_, columns=["infer_grade_remain"])
    df_roi = df_roi.groupby("img_name").sum()
    df_roi = df_roi.reset_index()
    img_name_grade = column_to_dict(df_img, "img_name", "grade")
    df_roi["grade"] = df_roi.img_name.apply(lambda x: img_name_grade[x])
    df_roi["grade"] = df_roi.grade.fillna(-1)
    df_roi["grade_text"] = df_roi.grade.apply(lambda x: GRADE_TEXT_DICT[x])

    del df_img_

    columns = ["img_name", "grade", "grade_text"]
    for g in GRADE_LIST:
        grade_c = f"infer_grade_remain_{g}"
        columns.append(grade_c)
        if grade_c not in df_roi.columns:
            df_roi[grade_c] = 0
    columns.extend(["infer_grade_remain_N_GRADE", "infer_grade_remain_N_GRADE_R"])
    df_roi = df_roi[columns]
    df_roi["grade_G1"] = (
        df_roi["infer_grade_remain_G1"] + df_roi["infer_grade_remain_G1-G2"] / 2
    )
    df_roi["grade_G2"] = (
        df_roi["infer_grade_remain_G2"]
        + df_roi["infer_grade_remain_G1-G2"] / 2
        + df_roi["infer_grade_remain_G2-G3"] / 2
    )
    df_roi["grade_G3"] = (
        df_roi["infer_grade_remain_G3"]
        + df_roi["infer_grade_remain_G2-G3"] / 2
        + df_roi["infer_grade_remain_G3-G4"] / 2
    )
    df_roi["grade_G4"] = (
        df_roi["infer_grade_remain_G4"] + df_roi["infer_grade_remain_G3-G4"] / 2
    )
    columns = list(df_roi.columns)[:-4]
    columns.extend(["grade_G4", "grade_G3", "grade_G2", "grade_G1"])
    df_roi = df_roi[columns]
    df_roi["infer_grade"] = df_roi.apply(apply_infer_grade_rate, axis=1)
    output = df_roi[["img_name", "infer_grade"]]
    output.to_csv(f"./result/clusters/{Config.folder_name}_infer_grade.csv")
    if Config.log:
        logging.debug(f"infer grade filename : {Config.folder_name}_infer_grade.csv")

    del df_roi, output

    acc = []
    result_val = df_img.groupby("img_name").sum()[["label", "result", "num"]]
    result_val.result = result_val.result / result_val.num * 64
    result_val.label = result_val.label.apply(lambda x: 1 if x > 0 else 0)
    for th in range(64):
        result_label = result_val.result.apply(lambda x: 1 if x > th else 0)
        acc.append(accuracy_score(result_val.label, result_label))
    th_all_cluster = np.argmax(acc)
    del result_val

    acc = []
    result_val = (
        df_img[df_img.remove == 0].groupby("img_name").sum()[["label", "result", "num"]]
    )
    result_val.result = result_val.result / result_val.num * 64
    result_val.label = result_val.label.apply(lambda x: 1 if x > 0 else 0)
    for th in range(64):
        result_label = result_val.result.apply(lambda x: 1 if x > th else 0)
        acc.append(accuracy_score(result_val.label, result_label))
    th_select_cluster = np.argmax(acc)

    del df_img, df_cluster

    df_img = create_img_df(result_name=Config.folder_name, val=False)
    df_cluster = create_cluster_df(df_img)

    df_cluster["remove"] = df_cluster.cluster_name.apply(lambda x: select_cluster[x])
    df_cluster["infer_grade"] = df_cluster.cluster_name.apply(
        lambda x: grade_cluster[x]
    )
    df_img["remove"] = df_img.cluster_name.apply(lambda x: select_cluster[x])
    df_img["infer_grade"] = df_img.cluster_name.apply(lambda x: grade_cluster[x])
    df_img["place"] = df_img.img_name_f.apply(lambda x: int(x[: x.find("_")]))

    result_df = df_img.groupby("img_name").sum()[["label", "result", "num"]]
    result_df.result = result_df.result / result_df.num * 64
    result_df.label = result_df.label.apply(lambda x: 1 if x > 0 else 0)
    result_df["result_label"] = result_df.result.apply(
        lambda x: 1 if x > th_all_cluster else 0
    )

    if Config.log:
        logging.debug("When use all clsuters")
        logging.debug(
            "acc: {:.4f}".format(
                accuracy_score(result_df.label, result_df.result_label)
            )
        )
        logging.debug(
            "spe: {:.4f}".format(
                specificity_score(result_df.label, result_df.result_label)
            )
        )
        logging.debug(
            "sen: {:.4f}".format(recall_score(result_df.label, result_df.result_label))
        )
        logging.debug(
            "auc: {:.4f}".format(roc_auc_score(result_df.label, result_df.result_label))
        )

    print("When use all clsuters")
    print("acc: {:.4f}".format(accuracy_score(result_df.label, result_df.result_label)))
    print(
        "spe: {:.4f}".format(specificity_score(result_df.label, result_df.result_label))
    )
    print("sen: {:.4f}".format(recall_score(result_df.label, result_df.result_label)))
    print("auc: {:.4f}".format(roc_auc_score(result_df.label, result_df.result_label)))

    del result_df

    result_df = (
        df_img[df_img.remove == 0].groupby("img_name").sum()[["label", "result", "num"]]
    )
    result_df.result = result_df.result / result_df.num * 64
    result_df.label = result_df.label.apply(lambda x: 1 if x > 0 else 0)
    result_df["result_label"] = result_df.result.apply(
        lambda x: 1 if x > th_select_cluster else 0
    )

    if Config.log:
        logging.debug("When use contributed clsuters")
        logging.debug(
            "acc: {:.4f}".format(
                accuracy_score(result_df.label, result_df.result_label)
            )
        )
        logging.debug(
            "spe: {:.4f}".format(
                specificity_score(result_df.label, result_df.result_label)
            )
        )
        logging.debug(
            "sen: {:.4f}".format(recall_score(result_df.label, result_df.result_label))
        )
        logging.debug(
            "auc: {:.4f}".format(roc_auc_score(result_df.label, result_df.result_label))
        )

        logging.debug("Number of images and clusters")
        logging.debug(f" All Clsuters :  {len(df_img)}, {len(df_cluster)}")
        logging.debug(
            f" Selected CLusters :  {len(df_img[df_img.remove == 0])}, {len(df_cluster[df_cluster.remove == 0])}"
        )

    print("When use contributed clsuters")
    print("acc: {:.4f}".format(accuracy_score(result_df.label, result_df.result_label)))
    print(
        "spe: {:.4f}".format(specificity_score(result_df.label, result_df.result_label))
    )
    print("sen: {:.4f}".format(recall_score(result_df.label, result_df.result_label)))
    print("auc: {:.4f}".format(roc_auc_score(result_df.label, result_df.result_label)))

    print("Number of images and clusters")
    print(f" All Clsuters :  {len(df_img)}, {len(df_cluster)}")
    print(
        f" Selected CLusters :  {len(df_img[df_img.remove == 0])}, {len(df_cluster[df_cluster.remove == 0])}"
    )
    if Config.log:
        logging.debug("Clusters with more than 90% of surgery or biopsy images")
    print("Clusters with more than 90% of surgery or biopsy images")
    buff = df_cluster[((df_cluster.surgery_rate > 0.9) & (df_cluster.remove == 0))]
    if Config.log:
        logging.debug(f" Surgery : {buff.num.sum()}, {len(buff)}")
    print(f" Surgery : {buff.num.sum()}, {len(buff)}")

    buff = df_cluster[((df_cluster.surgery_rate < 0.1) & (df_cluster.remove == 0))]
    if Config.log:
        logging.debug(f" Biopsy : {buff.num.sum()}, {len(buff)}")
    print(f" Biopsy : {buff.num.sum()}, {len(buff)}")

    buff = df_cluster[
        (
            (df_cluster.surgery_rate <= 0.9)
            & (df_cluster.remove == 0)
            & (df_cluster.surgery_rate >= 0.1)
        )
    ]
    if Config.log:
        logging.debug("Clusters with surgery and biopsy images")
        logging.debug(f" Coexistence : {buff.num.sum()}, {len(buff)}")

        logging.debug("CLusters with cancer or background")
        logging.debug(
            f" Cancer  : {df_cluster[((df_cluster.label == 1) & (df_cluster.remove == 0))].num.sum()}, {len(df_cluster[((df_cluster.label == 1) & (df_cluster.remove == 0))])}"
        )
        logging.debug(
            f" Background : {df_cluster[((df_cluster.label == 0) & (df_cluster.remove == 0))].num.sum()}, {len(df_cluster[((df_cluster.label == 0) & (df_cluster.remove == 0))])}"
        )

    print("Clusters with surgery and biopsy images")
    print(f" Coexistence : {buff.num.sum()}, {len(buff)}")
    print("CLusters with cancer or background")
    print(
        f" Cancer  : {df_cluster[((df_cluster.label == 1) & (df_cluster.remove == 0))].num.sum()}, {len(df_cluster[((df_cluster.label == 1) & (df_cluster.remove == 0))])}"
    )
    print(
        f" Background : {df_cluster[((df_cluster.label == 0) & (df_cluster.remove == 0))].num.sum()}, {len(df_cluster[((df_cluster.label == 0) & (df_cluster.remove == 0))])}"
    )

    result_df = result_df[["label", "result_label"]]
    result_df.to_csv(f"./result/{Config.folder_name}_result_label.csv")

    if Config.log:
        logging.debug(f"result filename : {Config.folder_name}_result_label.csv")
