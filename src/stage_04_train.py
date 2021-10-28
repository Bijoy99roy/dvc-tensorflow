from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import load_full_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
import argparse
import os
import logging



logging_str = r"[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs.log"), level=logging.INFO, format=logging_str, filemode="a")


def train_model(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['artifacts_dir']

    train_model_dir_path = os.path.join(artifacts_dir, artifacts['trained_model_dir'])

    create_directory([train_model_dir_path])

    untrained_full_model_path = os.path.join(
        artifacts_dir, 
        artifacts['base_model_dir'], 
        artifacts['updated_base_model_name'])

    model = load_full_model(untrained_full_model_path)

    callback_dir_path = os.path.join(artifacts_dir, artifacts['callbacks_dir'])
    callbacks = get_callbacks(callback_dir_path)

    train_generator, valid_generator = train_valid_generator(
        data_dir = artifacts['data_dir'],
        IMAGE_SIZE=tuple(params['image_size'][:-1]),
        BATCH_SIZE=params['batch_size'],
        do_data_augmentation=params['augmentation']
    )

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size
    
    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=params['epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()
    try:
        logging.info(">>>>>> stage four started")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage four completed! training completed and model saved >>>>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e