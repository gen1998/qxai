BASE_ROI_IMAGE_PATH = "./dataset/roi/"
BASE_PATCH_IMAGE_PATH = "./dataset/patch/"
BASE_HOVERNET_JSON_PATH = "./dataset/hovernet"

REPLACE_DICT = {
    "1～2": 1.5,
    "2～3": 2.5,
    "2‐3": 2.5,
    "3～4": 3.5,
    "4+3(Gleason風表記)": 3.5,
    "1-2": 1.5,
    "2-3": 2.5,
    "1+2": 1.5,
    "2+1": 1.5,
    "4+3": 3.5,
    "3(2にはしたくない)": 2.5,
    "(2)": 2,
    "1-1.5-2.5": 1.5,
    "0.5-1": 1,
}
GRADE_LIST = ["g_1", "g_1.5", "g_2", "g_2.5", "g_3", "g_3.5", "g_4"]
TXT_GRADE_DICT = {
    "g_1": "G1",
    "g_2": "G2",
    "g_3": "G3",
    "g_4": "G4",
    "g_12": "G1-G2",
    "g_23": "G2-G3",
    "g_34": "G3-G4",
}
