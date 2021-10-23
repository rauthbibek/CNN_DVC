from os import read
from src.utils.all_utils import read_yaml, create_dirs
from src.utils.models import get_VGG16_model, prepare_model
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

def prepare_callbacks(config_path, param_path):
    pass


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>>> Stage 3 started")
        prepare_callbacks(config_path=parsed_args.config, param_path=parsed_args.params)
        logging.info("Stage 3 completed and callbacks saved as a binary file >>>>>>>>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
