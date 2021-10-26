from os import read
from src.utils.all_utils import read_yaml, create_dirs
from src.utils.callbacks import create_and_save_tb_callback,create_and_save_ckpt_callback 
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

def prepare_callbacks(config_path):
    config = read_yaml(config_path)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']

    tb_log_dir = os.path.join(artifacts_dir, artifacts['TENSORBOARD_ROOT_LOG_DIR'])
    ckpt_dir = os.path.join(artifacts_dir, artifacts['CHECKPOINT_DIR'])
    callback_dir = os.path.join(artifacts_dir, artifacts['CALLBACKS_DIR'])

    create_dirs([tb_log_dir,
    ckpt_dir,callback_dir])

    create_and_save_tb_callback(callback_dir, tb_log_dir)
    create_and_save_ckpt_callback(callback_dir, ckpt_dir)




if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>>> Stage 3 started")
        prepare_callbacks(config_path=parsed_args.config)
        logging.info("Stage 3 completed and callbacks saved as a binary file >>>>>>>>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
