import tensorflow as tf
from .config import *

def initialize_optimizers(models, resume_epoch):

    """
    Initializes the optimizers for the Generator and Discriminator models.
    
    If training is resumed, the optimizers are built to ensure compatibility with 
    the model's trainable variables.
    
    Args:
        models (dict): Dictionary containing 'generator' and 'discriminator' models.
        resume_epoch (int or None): The epoch to resume training from. If None, training starts from scratch.
    
    Returns:
        dict: A dictionary containing the 'generator' and 'discriminator' optimizers.
    """

    optimizers = {
        'generator': tf.keras.optimizers.Adam(generator_learning_rate),
        'discriminator': tf.keras.optimizers.Adam(discriminator_learning_rate)
    }
    
    if resume_epoch:
        optimizers['generator'].build(models['generator'].trainable_variables)
        optimizers['discriminator'].build(models['discriminator'].trainable_variables)

    return optimizers

