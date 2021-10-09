from os import read
from src.utils.all_utils import read_yaml, create_dirs
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

def copy_file(source_download_dir, local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    N = len(list_of_files)
    for file in tqdm(list_of_files, total=N, desc=f"Copying files from {source_download_dir} to {local_data_dir}", colour="green"):
        src = os.path.join(source_download_dir, file)
        dst = os.path.join(local_data_dir, file)
        shutil.copy(src, dst)


def get_data(config_path):

    config = read_yaml(config_path)
    
    # save data in the local directory
    source_download_dirs = config["source_download_dirs"]
    local_data_dirs = config["local_data_dirs"]
    
    for source_download_dir, local_data_dir in tqdm(zip(source_download_dirs, local_data_dirs), total=2, desc="List of folders", colour="red"):
        create_dirs(dirs= [local_data_dir])
        copy_file(source_download_dir, local_data_dir)




if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>>> Stage 1 started")
        get_data(config_path=parsed_args.config)
        logging.info("Stage 1 completed and data saved in local file >>>>>>>>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e

