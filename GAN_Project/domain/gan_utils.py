import os
import matplotlib.pyplot as plt
import tensorflow as tf
from .config import *

def save_images(epoch, models):

    """
    Generates and saves a grid of images produced by the Generator.
    
    Args:
        epoch (int): The current training epoch.
        models (dict): Dictionary containing 'generator' and 'discriminator' models.
    """

    generator = models['generator']

    fig, axes = plt.subplots(4, 4, figsize=(6, 7))
    fig.subplots_adjust(wspace=0.02, hspace=0.01)

    noise = tf.random.normal([16, latent_dim]) 
    generated_images = generator(noise, training=False)  

    for i, ax in enumerate(axes.flat):
        generated_image = generated_images[i]
        
        generated_image = (generated_image + 1) / 2.0 
        generated_image = (generated_image * 255).numpy().astype("uint8") 
        
        ax.imshow(generated_image)
        ax.axis("off")  

    image_path = os.path.join(images_dir, f"epoch_{epoch + 1}.png")
    fig.savefig(image_path, bbox_inches="tight", dpi=200)

def save_model(epoch, models):

    """
    Saves the Generator and Discriminator models to the models directory.
    
    Args:
        epoch (int): The current training epoch.
        models (dict): Dictionary containing 'generator' and 'discriminator' models.
    """

    generator = models['generator']
    discriminator = models['discriminator']

    generator_path = os.path.join(models_dir, f"generator_epoch_{epoch + 1}.keras")
    generator.save(generator_path)
    discriminator_path = os.path.join(models_dir, f"discriminator_epoch_{epoch + 1}.keras")
    discriminator.save(discriminator_path)

