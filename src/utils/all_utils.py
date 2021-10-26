import yaml
import os
import json
import logging
import time

def read_yaml(path_to_yaml:str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml  file: {path_to_yaml} loaded successfully")
    return content

def create_dirs(dirs: list):

    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        logging.info(f"Directory is created at {dir}")

def save_local_df(data, path):

    data.to_csv(path, index=False )
    logging.info(f"data saved at {path}")

def save_scores(scores, path, indentation=4):
    with open(path, "w") as f:
        json.dump(scores, f, indent=indentation)
    logging.info(f"scores are saved at {path}")

def get_timestamp(name: str):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_") 
    unique_name = f"{name}_at_{timestamp}"
    return unique_name


