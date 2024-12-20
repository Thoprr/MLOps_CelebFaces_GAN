import tensorflow as tf

# Loss function used for both generator and discriminator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):

    """
    Calculates the discriminator loss for real and fake outputs.
    
    Args:
        real_output (tf.Tensor): Discriminator predictions for real images.
        fake_output (tf.Tensor): Discriminator predictions for generated (fake) images.
    
    Returns:
        tf.Tensor: Total loss for the discriminator.
    """

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  # Real images labeled as 1
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  # Fake images labeled as 0 
    total_loss = real_loss + fake_loss
    
    return total_loss

def generator_loss(fake_output):

    """
    Calculates the generator loss based on the discriminator's feedback.
    
    Args:
        fake_output (tf.Tensor): Discriminator predictions for generated (fake) images.
    
    Returns:
        tf.Tensor: Loss for the generator.
    """

    return cross_entropy(tf.ones_like(fake_output), fake_output) # Fake images should be classified as real (1)