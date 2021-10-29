from os import read
from src.utils.all_utils import read_yaml, create_dirs
from src.utils.models import load_full_model, get_unique_model_name
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
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

def train_model(config_path, param_path):
    config = read_yaml(config_path)
    params = read_yaml(param_path)
    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']

    untrained_full_modelpath=os.path.join(artifacts_dir,artifacts['BASE_MODEL_DIR'], artifacts['CUSTOM_MODEL_NAME'])

    model = load_full_model(untrained_full_modelpath)

    callback_dir_path = os.path.join(artifacts_dir,artifacts['CALLBACKS_DIR'])
    callbacks = get_callbacks(callback_dir_path)

    train_generator, validation_generator = train_valid_generator(
        data_dir=artifacts['DATA_DIR'],
        IMAGE_SIZE=tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE=params["BATCH_SIZE"],
        do_data_augmentation=params["AUGMENTATION"]
    )

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size
    logging.info(f"Model training started")

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs = params["EPOCHS"],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    ) 

    logging.info(f"Model training completed")

    trained_model_dir = os.path.join(artifacts_dir,artifacts['TRAINED_MODEL_DIR'])
    create_dirs([trained_model_dir])

    model_file_path = os.path.join(trained_model_dir, get_unique_model_name())
    model.save(model_file_path)

    logging.info(f"Model is saved at {model_file_path}")





if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>>> Stage 4 started")
        train_model(config_path=parsed_args.config, param_path=parsed_args.params)
        logging.info("Stage 4 completed and training completed >>>>>>>>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
