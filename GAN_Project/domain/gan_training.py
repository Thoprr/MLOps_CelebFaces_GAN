from time import time
import numpy as np
import tensorflow as tf
from .config import *
from .data_loader import load_and_preprocess_dataset
from .gan_models import initialize_models
from .gan_losses import generator_loss, discriminator_loss
from .gan_optimizers import initialize_optimizers
from .gan_logger import create_log_file, log_epoch_status
from .gan_utils import save_images, save_model

@tf.function
def train_generator_step(models, optimizers, noise):

    """
    Performs a single training step for the Generator.

    Args:
        models (dict): Dictionary containing 'generator' and 'discriminator' models.
        optimizers (dict): Dictionary containing optimizers for generator and discriminator.
        noise (tf.Tensor): Random noise used as input for the generator.
    
    Returns:
        tf.Tensor: Loss for the generator.
    """

    generator_optimizer = optimizers['generator']
    generator = models['generator']
    discriminator = models['discriminator']

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss

@tf.function
def train_discriminator_step(models, optimizers, images, noise):

    """
    Performs a single training step for the Discriminator.

    Args:
        models (dict): Dictionary containing 'generator' and 'discriminator' models.
        optimizers (dict): Dictionary containing optimizers for generator and discriminator.
        images (tf.Tensor): Real images from the dataset.
        noise (tf.Tensor): Random noise used to generate fake images.
    
    Returns:
        tf.Tensor: Loss for the discriminator.
    """

    generator = models['generator']
    discriminator = models['discriminator']
    discriminator_optimizer = optimizers['discriminator']

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return disc_loss

def train_step(images, batch_size, models, train_flags, previous_losses, optimizers):

    """
    Executes one training step for both Generator and Discriminator.

    Args:
        images (tf.Tensor): Real images from the dataset.
        batch_size (int): Number of samples per batch.
        models (dict): Dictionary containing 'generator' and 'discriminator' models.
        train_flags (dict): Flags to determine if generator or discriminator should be trained.
        previous_losses (dict): Previous epoch's generator and discriminator loss.
        optimizers (dict): Dictionary containing optimizers for generator and discriminator.
    
    Returns:
        dict: Dictionary containing generator and discriminator losses.
    """

    noise = tf.random.normal([batch_size, latent_dim])
    
    losses = {
        'generator': train_generator_step(models, optimizers, noise) if train_flags['generator'] else previous_losses['generator'],
        'discriminator': train_discriminator_step(models, optimizers, images, noise) if train_flags['discriminator'] else previous_losses['discriminator']
    }

    return losses

def train_one_epoch(dataset, batch_size, models, train_flags, previous_losses, optimizers):

    """
    Trains the Generator and Discriminator for one epoch.

    Args:
        dataset (tf.data.Dataset): The dataset of real images.
        batch_size (int): Number of samples per batch.
        models (dict): Dictionary containing 'generator' and 'discriminator' models.
        train_flags (dict): Flags to determine if generator or discriminator should be trained.
        previous_losses (dict): Previous epoch's generator and discriminator loss.
        optimizers (dict): Dictionary containing optimizers for generator and discriminator.
    
    Returns:
        tuple: (average losses for generator and discriminator, last batch losses)
    """

    losses_total = {
        'generator': 0,
        'discriminator': 0
    }
    batch_count = 0

    threshold = train_ratio_threshold

    for image_batch in dataset:
        losses = train_step(image_batch, batch_size, models, train_flags, previous_losses, optimizers)
        previous_losses = losses

        gen_loss_np, disc_loss_np = (loss.numpy() for loss in (losses['generator'], losses['discriminator']))
        epsilon = 1e-8
        train_ratio = (gen_loss_np - disc_loss_np) / max(gen_loss_np, disc_loss_np, epsilon)

        if train_ratio > threshold:
            train_flags['generator'] = True
            train_flags['discriminator'] = False
        elif train_ratio < -threshold:
            train_flags['generator'] = False
            train_flags['discriminator'] = True
        else:
            train_flags['generator'] = True
            train_flags['discriminator'] = True

        losses_total['generator'] += gen_loss_np
        losses_total['discriminator'] += disc_loss_np
        batch_count += 1

    losses_avg = {key: total / batch_count for key, total in losses_total.items()}

    return losses_avg, previous_losses

def train_gan(epochs, batch_size, resume_epoch=None):

    """
    Trains the GAN model for a specified number of epochs.

    Args:
        epochs (int): Total number of epochs to train.
        batch_size (int): Number of samples per batch.
        resume_epoch (int or None): The epoch to resume training from. If None, training starts from scratch.
    
    Returns:
        None
    """

    models = initialize_models(resume_epoch)

    train_flags = {'generator': True, 'discriminator': True}
    
    last_losses = {}

    optimizers = initialize_optimizers(models, resume_epoch)

    dataset = load_and_preprocess_dataset(batch_size)

    log_file = create_log_file()

    start_epoch = resume_epoch if resume_epoch is not None else 0

    for epoch in range(start_epoch, start_epoch + epochs):  
        start = time()

        loss_avg, last_losses = train_one_epoch(
            dataset, batch_size, models, train_flags, last_losses, optimizers
        ) 

        epoch_duration = time() - start
        log_epoch_status(log_file, epoch, epoch_duration, loss_avg)

        if (epoch+1) % models_save_interval == 0:
            save_model(epoch, models)

        if (epoch+1) % images_save_interval == 0:
            save_images(epoch, models)