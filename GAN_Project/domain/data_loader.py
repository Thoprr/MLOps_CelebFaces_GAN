import os
import tensorflow as tf
from .config import *

def load_and_preprocess_dataset(batch_size):

    """
    Loads and preprocesses an image dataset from the directory specified in the configuration.

    Args:
        batch_size (int): The number of images per batch.

    Returns:
        tf.data.Dataset: A preprocessed and batched TensorFlow dataset.
    """

    image_files = [f for f in os.listdir(processed_data_dir) if os.path.isfile(os.path.join(processed_data_dir, f))]
    if len(image_files) == 0:
        raise ValueError(f"No images found in '{processed_data_dir}'.")
    
    print(f"Directory '{processed_data_dir}' contains {len(image_files)} files.")

    # Load dataset from directory
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        processed_data_dir,
        batch_size=batch_size,
        label_mode=None,  
        shuffle=True  
    )

    # Resize and normalize images
    def preprocess_image(image):
        image = tf.image.resize(image, tuple(image_size))
        image = tf.cast(image, tf.float32)  
        image = (image / 127.5) - 1 # Normalize to [-1, 1]
        return image

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset