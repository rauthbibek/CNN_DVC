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



def base_model(config_path, param_path):

    config = read_yaml(config_path)
    params = read_yaml(param_path)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    base_model_dir = artifacts['BASE_MODEL_DIR']
    base_model_name = artifacts['BASE_MODEL_NAME']
    base_model_dir_path =  os.path.join(artifacts_dir, base_model_dir)
    create_dirs([base_model_dir_path])
    base_model_path = os.path.join(base_model_dir_path, base_model_name)
    input_shape = params['IMAGE_SIZE']
    learning_rate = params['LEARNING_RATE']
    classes = params['N_CLASS']

    model = get_VGG16_model(input_shape=input_shape, model_path=base_model_path)

    full_model = prepare_model(model, 
    classes, 
    freeze_all=True, 
    freeze_till=None,
    learning_rate=learning_rate)

    custom_model_name = artifacts['CUSTOM_MODEL_NAME']
    custom_model_path = os.path.join(base_model_dir_path,custom_model_name)
    def _log_model_summary(full_model):
        with io.StringIO() as stream:
            full_model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    logging.info(f"full model summary: \n{_log_model_summary(full_model)}")

    full_model.save(custom_model_path)

    
 


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>>> Stage 2 started")
        base_model(config_path=parsed_args.config, param_path=parsed_args.params)
        logging.info("Stage 2 completed and model saved >>>>>>>>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e

