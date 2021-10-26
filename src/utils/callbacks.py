import tensorflow as tf
import os
import joblib
import logging
from src.utils.all_utils import get_timestamp


def create_and_save_tb_callback(callback_dir, tb_log_dir):
    unique_name = get_timestamp("tb_logs")

    tb_running_logs_dir = os.path.join(tb_log_dir, unique_name)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_running_logs_dir)

    tb_callback_filepath = os.path.join(callback_dir, "tensorboard_cb.cb")
    joblib.dump(tb_callback, tb_callback_filepath)
    logging.info(f"Tensorboard callback saved at {tb_callback_filepath}")



def create_and_save_ckpt_callback(callback_dir, ckpt_dir):
    
    ckpt_file = os.path.join(ckpt_dir, "ckpt_model.h5")
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_file,
        save_best_only=True
    )

    ckpt_callback_filepath = os.path.join(callback_dir, "checkpoint_cb.cb")
    joblib.dump(ckpt_callback, ckpt_callback_filepath)
    logging.info(f"Checkpoint callback saved at {ckpt_callback_filepath}")
