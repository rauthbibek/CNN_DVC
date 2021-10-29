import tensorflow as tf
import logging
from src.utils.all_utils import get_timestamp

#preparing base model
def get_VGG16_model(input_shape, model_path):
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False
    )

    model.save(model_path)
    logging.info(f"VGG16 model saved at {model_path}")
    return model

#adding custom layers based on our requirements
def prepare_model(model, CLASSES, freeze_all, freeze_till, learning_rate):
    if freeze_all:
        for layer in model.layers:
            layer.trainable =  False
        logging.info("Freeze all layers are done")
    elif (freeze_till is not None) and (freeze_till>0):
        for layer in model.layers[:-freeze_till]:
            layer.trainable =  False
        logging.info("Last {freeze_till} layer(s) are left for training and others are freeze")

    flatten_in = tf.keras.layers.Flatten()(model.output)
    predictions = tf.keras.layers.Dense(units=CLASSES,
    activation='softmax')(flatten_in)

    full_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=predictions
    )

    full_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
    )

    logging.info("custom model compiled and ready to be trained")
    full_model.summary()
    return full_model


def load_full_model(untrained_full_modelpath):

    model = tf.keras.models.load_model(untrained_full_modelpath)
    logging.info(f"Untrained model is returned from: {untrained_full_modelpath}")

    return model

def get_unique_model_name(model_name='model'):
    unique_name = get_timestamp(model_name)
    unique_model_name = f"{unique_name}_.h5"

    return unique_model_name




