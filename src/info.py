import numpy as np
import pandas as pd

from src.utils import column_to_dict

from . import GRADE_LIST


# apply
def apply_cluster_name(row):
    try:
        k = row.index[np.where(row.values == -100)[0][0] - 1]
        label = int(k[k.find("_") + 1 :])
        index = row[k]
    except:
        label = 59
        index = row["labels_59"]

    try:
        cl_name = f"{row.fold}_{label}_{index}"
    except:
        cl_name = f"{label}_{index}"
    return cl_name


def apply_img_name_f(row):
    return row[row.rfind("/") + 1 :]


def apply_correct(row):
    label = row.label
    infer = row.infer

    if label == infer:
        return 1
    else:
        return 0


def extract_result(project_name, val=True, fold=-1):
    df_result = pd.read_csv(f"./result/predictions/{project_name}_val_preds.csv")
    if fold > -1:
        df_result = df_result[df_result.fold == fold + 1]
    if val:
        df_result = df_result[df_result.ver == "val"]
    else:
        df_result = df_result[df_result.ver == "test"]

    df_result["img_name_f"] = df_result["img_path"].apply(apply_img_name_f)
    df_result["img_name"] = df_result.img_name_f.apply(lambda x: x[x.find("_") + 1 :])
    df_result["num"] = 1
    df_result["result"] = 1 - df_result["result"]
    df_result["surgery"] = df_result.img_name.apply(lambda x: 0 if "tif" in x else 1)
    df_result = df_result.reset_index(drop=True)

    return df_result


def create_img_df(project_name, val=True, fold=-1):
    if fold > -1:
        df = pd.read_csv(f"./result/predicitons/{project_name}_kmeans.csv")
    else:
        df = pd.read_csv(f"./data/output/result/{project_name}_kmeans_{fold + 1}.csv")

    if val:
        df = df[df.ver == "val"].reset_index(drop=True)
    else:
        df = df[df.ver == "test"].reset_index(drop=True)

    df["img_name"] = df.img_name_f.apply(lambda x: x[x.find("_") + 1 :])
    df["cluster_name"] = df.apply(apply_cluster_name, axis=1)

    # input grade
    not_exist_grade = df[
        ((df.label == 0) | (df.category == "biopsy"))
    ].img_name.unique()
    df_grade = pd.read_csv("./dataset/csv/input.csv")
    df_grade = df_grade[
        ((df_grade.label == 1) & (df_grade.category == "surgery"))
    ].reset_index(drop=True)
    df_grade.grade = df_grade.grade.astype(np.float64)
    dict_grade = column_to_dict(df_grade, "img_name", "grade")
    for n in not_exist_grade:
        dict_grade[n] = -1
    df["grade"] = df.img_name.apply(lambda x: dict_grade[x])

    # input inference
    df_result = extract_result(project_name=project_name, val=val, fold=fold)
    dict_result = column_to_dict(df_result, "img_name_f", "result")
    df["result"] = df.img_name_f.apply(lambda x: dict_result[x])
    df["infer"] = df.result.apply(lambda x: 0 if x < 0.5 else 1)
    df["surgery"] = df.category.apply(lambda x: 0 if x == "biopsy" else 1)
    df["correct"] = df.apply(apply_correct, axis=1)

    df = df[
        [
            "img_name_f",
            "img_name",
            "label",
            "category",
            "grade",
            "cluster_name",
            "infer",
            "result",
            "surgery",
            "correct",
        ]
    ]

    return df


def create_cluster_df(df):
    dict_v = df.groupby("cluster_name").grade.value_counts().to_dict()
    n_list = df.cluster_name.unique()

    dict_value = {}

    for name in n_list:
        output = [0] * 7
        for index, v in enumerate([1, 1.5, 2, 2.5, 3, 3.5, 4]):
            try:
                output[index] = dict_v[(name, v)]
            except:
                pass

        dict_value[name] = output

    grade = pd.DataFrame(dict_value, index=GRADE_LIST).T

    df["surgery"] = df.category.apply(lambda x: 0 if x == "biopsy" else 1)
    df["num"] = 1
    df["correct"] = df.apply(apply_correct, axis=1)

    df = df.groupby("cluster_name").sum()[["num", "surgery", "label", "correct"]]
    df["biopsy"] = df.num - df.surgery
    df["label_rate"] = df.label / df.num
    df = df.drop(["label"], axis=1)
    df["label"] = df.label_rate.apply(lambda x: 0 if x < 0.5 else 1)

    df["correct_rate"] = df["correct"] / df["num"]
    df["surgery_rate"] = df["surgery"] / df["num"]

    df = df[
        ["label", "num", "correct", "correct_rate", "surgery", "biopsy", "surgery_rate"]
    ]

    df = df.merge(grade, left_on="cluster_name", right_index=True).reset_index()

    df["g_1_p"] = df["g_1"] + df["g_1.5"] / 2
    df["g_2_p"] = df["g_2"] + df["g_1.5"] / 2 + df["g_2.5"] / 2
    df["g_3_p"] = df["g_3"] + df["g_2.5"] / 2 + df["g_3.5"] / 2
    df["g_4_p"] = df["g_4"] + df["g_3.5"] / 2

    for c_name in ["g_1", "g_2", "g_3", "g_4"]:
        df[f"{c_name}_p_per"] = df[f"{c_name}_p"] / df.loc[:, "g_1_p":"g_4_p"].sum(
            axis=1
        )

    sum_grade = df.loc[:, "g_1_p":"g_4_p"].sum()
    df["g_1_p_st"] = df["g_1_p"] / sum_grade["g_1_p"]
    df["g_2_p_st"] = df["g_2_p"] / sum_grade["g_2_p"]
    df["g_3_p_st"] = df["g_3_p"] / sum_grade["g_3_p"]
    df["g_4_p_st"] = df["g_4_p"] / sum_grade["g_4_p"]

    df["g_12_p_st"] = df["g_1_p_st"] + df["g_2_p_st"]
    df["g_23_p_st"] = df["g_2_p_st"] + df["g_3_p_st"]
    df["g_34_p_st"] = df["g_3_p_st"] + df["g_4_p_st"]

    sum_grade = df.loc[:, "g_1_p_st":"g_4_p_st"].sum(axis=1)
    for c_name in ["1", "2", "3", "4", "12", "23", "34"]:
        df[f"g_{c_name}_p_st_per"] = df[f"g_{c_name}_p_st"] / sum_grade

    return df
