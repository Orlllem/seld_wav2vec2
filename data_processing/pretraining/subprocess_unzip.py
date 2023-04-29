import os
import shutil

import yaml

params = yaml.safe_load(open("params.yaml"))["unzip_pret_data"]


ROOT_DIR = os.path.abspath(os.curdir)

os.environ["DATA_PATH"] = f"{ROOT_DIR}/{params['data_path']}"
os.environ["DATA_WORK_PATH"] = f"{ROOT_DIR}/{params['data_work_path']}"

if os.path.isdir(params["data_work_path"]):
    shutil.rmtree(params["data_work_path"])
os.makedirs(params["data_work_path"])

os.system("sh data_processing/pretraining/unzip_pretraining_datasets.sh")
