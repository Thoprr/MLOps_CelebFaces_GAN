import os
import json

# 1️⃣ Load configuration file
config_path = os.path.join('..', 'config', 'config.json')
with open(config_path, 'r') as file:
    CONFIG = json.load(file)

# 2️⃣ Create necessary directories
for path in CONFIG['PATHS'].values():
    os.makedirs(path, exist_ok=True)

# 3️⃣ Extract paths
raw_data_dir = CONFIG['PATHS']['RAW_DATA']
additional_raw_data_dir = CONFIG['PATHS']['ADDITIONAL_RAW_DATA']
processed_data_dir = CONFIG['PATHS']['PROCESSED_DATA']
models_dir = CONFIG['PATHS']['MODELS_DIR']
images_dir = CONFIG['PATHS']['IMAGES_DIR']
logs_dir = CONFIG['PATHS']['LOGS']

# 4️⃣ Extract save intervals
models_save_interval = CONFIG['SAVE_INTERVALS']['MODELS']
images_save_interval = CONFIG['SAVE_INTERVALS']['IMAGES']

# 5️⃣ Extract training parameters
image_size = CONFIG['TRAINING']['IMAGE_SIZE']
latent_dim = CONFIG['TRAINING']['LATENT_DIM']
train_ratio_threshold = CONFIG['TRAINING']['TRAIN_RATIO_THRESHOLD']

# 6️⃣ Extract learning rates
generator_learning_rate = CONFIG['TRAINING']['LEARNING_RATES']['GENERATOR']
discriminator_learning_rate = CONFIG['TRAINING']['LEARNING_RATES']['DISCRIMINATOR']