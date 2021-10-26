from os import read
from src.utils.all_utils import read_yaml, create_dirs
import argparse
import pandas as pd
import os
from tqdm import tqdm
import logging
import io

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

def train_model(config_path):
    config = read_yaml(config_path)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>>> Stage 4 started")
        train_model(config_path=parsed_args.config)
        logging.info("Stage 4 completed and training completed >>>>>>>>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
