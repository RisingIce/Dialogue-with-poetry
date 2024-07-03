import sys, os

import torch


# 推理用的指定模型
sovits_path = r"app\core\SoVITS_weights\kafuka_e15_s750.pth"
gpt_path = r"app\core\GPT_weights\kafuka-e10.ckpt"


cnhubert_path = "app\core\GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "app\core\GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
pretrained_sovits_path = "app\core\GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_path = "app\core\GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

exp_root = "logs"
python_exec = sys.executable or "python"

if torch.cuda.is_available():
    infer_device = "cuda"
elif torch.backends.mps.is_available():
    infer_device = "mps"
else:
    infer_device = "cpu"


api_port = 9880
bind_addr = "127.0.0.1"

if infer_device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if (
        ("16" in gpu_name and "V100" not in gpu_name.upper())
        or "P40" in gpu_name.upper()
        or "P10" in gpu_name.upper()
        or "1060" in gpu_name
        or "1070" in gpu_name
        or "1080" in gpu_name
    ):
        is_half = False

dict_language = {
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "ZH": "zh",
    "EN": "en",
    "JA": "ja",
    "zh": "zh",
    "en": "en",
    "ja": "ja",
}

streaming = True

base_url = "https://jiuge.thunlp.org/jiugepoem/task/"

pairs_base_url = "https://jiuge.thunlp.org/"

import pandas as pd

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "json": pd.read_json,
    "html": pd.read_html,
    "xml": pd.read_xml,
    "pickle": pd.read_pickle,
}
