import os
import re
from PIL import Image
import tensorflow as tf
from .config import *
from .gan_generator import Generator

def get_latest_generator_model():

    """
    Returns the path of the latest generator model file.

    Searches for files matching the pattern 'generator_epoch_XXXX.keras' in the models directory, 
    sorts them by epoch number, and returns the path to the latest model.

    Returns:
        str: Path to the latest generator model file.

    Raises:
        FileNotFoundError: If no matching model files are found in the models directory.
    """

    model_files = [f for f in os.listdir(models_dir) if re.match(r'generator_epoch_\d+\.keras', f)]

    if not model_files:
        raise FileNotFoundError(f"Aucun modèle trouvé dans le dossier {models_dir}.")
    model_files.sort(key=lambda x: int(re.search(r'\d+', x).group())) 

    return os.path.join(models_dir, model_files[-1])  


def load_generator_model(model_api):

    """
    Loads a generator model from the specified file or the latest available model.

    If a model name is provided, it loads the specified model. 
    Otherwise, it loads the latest available model from the models directory.

    Args:
        model_api (str): Name of the model file to load. If empty, loads the latest model.

    Returns:
        tf.keras.Model: The loaded generator model.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
    """

    if model_api:  
        model_path = os.path.join(models_dir, model_api)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Le modèle spécifié {model_path} n'existe pas.")
    else: 
        model_path = get_latest_generator_model()
    
    print(f"Chargement du modèle : {model_path}")
    generator = tf.keras.models.load_model(model_path)

    return generator


def generate_image(generator, output):

    """
    Generates an image using the generator model and saves it as a PNG file.

    A random noise vector is used as input for the generator to produce an image. 
    The generated image is rescaled to the 0-255 range and saved as a PNG file.

    Args:
        generator (tf.keras.Model): The generator model used to generate the image.
        output (str): Path to save the generated image file.

    Returns:
        None
    """

    noise = tf.random.normal([1, latent_dim])
    generated_image = generator(noise)
    generated_image = (generated_image[0] + 1) / 2.0  
    generated_image = (generated_image * 255).numpy().astype("uint8")  

    image = Image.fromarray(generated_image)
    image.save(output)