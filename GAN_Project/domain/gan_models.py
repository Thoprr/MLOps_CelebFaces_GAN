import os
import tensorflow as tf
from keras.saving import register_keras_serializable
from .config import *

@register_keras_serializable()
class Generator(tf.keras.Model):

    """
    The Generator model for a GAN, which takes a latent vector as input 
    and produces a synthetic image as output.
    """

    def __init__(self, **kwargs):

        super(Generator, self).__init__(**kwargs)

        self.Dense_inputs = tf.keras.layers.Dense(5*4*1024, use_bias=False)
        self.BatchNormalization_inputs = tf.keras.layers.BatchNormalization()
        self.LeakyReLU_inputs = tf.keras.layers.LeakyReLU()
        self.Reshape = tf.keras.layers.Reshape((5, 4, 1024))

        self.Conv2DTranspose_1 = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding="same", use_bias=False)
        self.BatchNormalization_1 = tf.keras.layers.BatchNormalization()
        self.LeakyReLU_1 = tf.keras.layers.LeakyReLU()

        self.Conv2DTranspose_2 = tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", use_bias=False)
        self.BatchNormalization_2 = tf.keras.layers.BatchNormalization()
        self.LeakyReLU_2 = tf.keras.layers.LeakyReLU()

        self.Conv2DTranspose_3 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False)
        self.BatchNormalization_3 = tf.keras.layers.BatchNormalization()
        self.LeakyReLU_3 = tf.keras.layers.LeakyReLU()

        self.Conv2DTranspose_4 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False)
        self.BatchNormalization_4 = tf.keras.layers.BatchNormalization()
        self.LeakyReLU_4 = tf.keras.layers.LeakyReLU()

        self.Conv2DTranspose_output= tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh")

    def call(self, inputs, training=False):

        x = self.Dense_inputs(inputs)
        x = self.BatchNormalization_inputs(x, training=training)
        x = self.LeakyReLU_inputs(x)
        x = self.Reshape(x)

        x = self.Conv2DTranspose_1(x)
        x = self.BatchNormalization_1(x, training=training)
        x = self.LeakyReLU_1(x)

        x = self.Conv2DTranspose_2(x)
        x = self.BatchNormalization_2(x, training=training)
        x = self.LeakyReLU_2(x)

        x = self.Conv2DTranspose_3(x)
        x = self.BatchNormalization_3(x, training=training)
        x = self.LeakyReLU_3(x)

        x = self.Conv2DTranspose_4(x)
        x = self.BatchNormalization_4(x, training=training)
        x = self.LeakyReLU_4(x)

        output = self.Conv2DTranspose_output(x)

        return output

    def get_config(self):
        config = super(Generator, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class Discriminator(tf.keras.Model):

    """
    The Discriminator model for a GAN, which takes an image as input 
    and outputs a prediction of whether the image is real or fake.
    """

    def __init__(self, **kwargs):

        super(Discriminator, self).__init__(**kwargs)
        
        self.Conv2_1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")
        self.LeakyReLU_1 = tf.keras.layers.LeakyReLU()
        self.Dropout_1 = tf.keras.layers.Dropout(0.3)

        self.Conv2_2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")
        self.LeakyReLU_2 = tf.keras.layers.LeakyReLU()
        self.Dropout_2 = tf.keras.layers.Dropout(0.3)

        self.Conv2_3 = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same")
        self.LeakyReLU_3 = tf.keras.layers.LeakyReLU()
        self.Dropout_3 = tf.keras.layers.Dropout(0.3)

        self.Conv2_4 = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same")
        self.LeakyReLU_4 = tf.keras.layers.LeakyReLU()
        self.Dropout_4 = tf.keras.layers.Dropout(0.3)    

        self.Flatten_output = tf.keras.layers.Flatten()
        self.Dense_output = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        
        x = self.Conv2_1(inputs)
        x = self.LeakyReLU_1(x)
        x = self.Dropout_1(x, training=training)

        x = self.Conv2_2(x)
        x = self.LeakyReLU_2(x)
        x = self.Dropout_2(x, training=training)

        x = self.Conv2_3(x)
        x = self.LeakyReLU_3(x)
        x = self.Dropout_3(x, training=training)

        x = self.Conv2_4(x)
        x = self.LeakyReLU_4(x)
        x = self.Dropout_4(x, training=training)

        x = self.Flatten_output(x)
        output = self.Dense_output(x)

        return output
    
    def get_config(self):
        config = super(Discriminator, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def initialize_models(resume_epoch):

    """
    Initializes the Generator and Discriminator models.
    
    If `resume_epoch` is provided, it attempts to load pre-trained models from the models directory.
    Otherwise, it creates new instances of Generator and Discriminator.
    
    Args:
        resume_epoch (int or None): The epoch to resume training from. If None, new models are created.
    
    Returns:
        dict: A dictionary containing the 'generator' and 'discriminator' models.
    """

    if resume_epoch is None:

        print("Starting new training session...")
        models = {
            'generator': Generator(),
            'discriminator': Discriminator()
        }
        
    else:

        print(f"Resuming training from epoch {resume_epoch}...")
        
        generator_path = os.path.join(f".\{models_dir}", f"generator_epoch_{resume_epoch}.keras")
        discriminator_path = os.path.join(f".\{models_dir}", f"discriminator_epoch_{resume_epoch}.keras")

        if os.path.exists(generator_path) and os.path.exists(discriminator_path):
            models = {
                'generator': tf.keras.models.load_model(generator_path),
                'discriminator': tf.keras.models.load_model(discriminator_path)
            }
            print(f"Models loaded from {generator_path} and {discriminator_path}")
        else:
            raise FileNotFoundError(f"Files {generator_path} and/or {discriminator_path} do not exist.")

    return models