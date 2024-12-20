import os
import json

config_path = os.path.join('..', 'config', 'config.json')

with open(config_path, 'r') as file:
    CONFIG = json.load(file)

for path in CONFIG['PATHS'].values():
    os.makedirs(path, exist_ok=True)

raw_data_dir = CONFIG['PATHS']['RAW_DATA']
additional_raw_data_dir = CONFIG['PATHS']['ADDITIONAL_RAW_DATA']
processed_data_dir = CONFIG['PATHS']['PROCESSED_DATA']
models_dir = CONFIG['PATHS']['MODELS_DIR']
images_dir = CONFIG['PATHS']['IMAGES_DIR']
logs_dir = CONFIG['PATHS']['LOGS']

models_save_interval = CONFIG['SAVE_INTERVALS']['MODELS']
images_save_interval = CONFIG['SAVE_INTERVALS']['IMAGES']

image_size = CONFIG['TRAINING']['IMAGE_SIZE']
latent_dim = CONFIG['TRAINING']['LATENT_DIM']
train_ratio_threshold = CONFIG['TRAINING']['TRAIN_RATIO_THRESHOLD']

generator_learning_rate = CONFIG['TRAINING']['LEARNING_RATES']['GENERATOR']
discriminator_learning_rate = CONFIG['TRAINING']['LEARNING_RATES']['DISCRIMINATOR']