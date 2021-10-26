from os import read
from src.utils.all_utils import read_yaml, create_dirs
from src.utils.models import load_full_model
from src.utils.callbacks import get_callbacks
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

    untrained_full_modelpath=os.path.join(artifacts_dir,artifacts['BASE_MODEL_DIR'], artifacts['CUSTOM_MODEL_NAME'])

    model = load_full_model(untrained_full_modelpath)

    callback_dir_path = os.path.join(artifacts_dir,artifacts['CALLBACKS_DIR'])
    callbacks = get_callbacks(callback_dir_path)




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
