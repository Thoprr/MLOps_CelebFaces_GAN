import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class Generator(tf.keras.Model):

    """
    Convolutional generator model for image synthesis.

    Transforms random noise into an image using transposed convolutions, 
    batch normalization, and LeakyReLU activations. The final layer outputs 
    an image with 3 channels and 'tanh' activation.

    Methods:
        call(inputs, training=False): Forward pass to generate an image from input noise.
        get_config(): Returns the model's configuration.
        from_config(config): Recreates the model from its configuration.
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