import os
import re
from PIL import Image, ImageDraw, ImageFont
from .config import *

def natural_sort_key(text):

    """
    Generates a natural sort key for a given text.

    Splits the text into segments of numbers and letters, converting numbers to integers 
    and lowercase letters for natural sorting.

    Args:
        text (str): The input text to be sorted.

    Returns:
        list: List of integers and lowercase strings for natural sorting.
    """

    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', text)]

def create_gif(gif_path, duration=200, font_size=50):

    """
    Creates a GIF from PNG images in a specified directory.

    Images are sorted naturally, annotated with the epoch number, and combined into a GIF.
    The resulting GIF is saved at the specified path.

    Args:
        gif_path (str): Path where the GIF will be saved.
        duration (int, optional): Duration of each frame in milliseconds. Default is 200ms.
        font_size (int, optional): Font size for epoch annotation. Default is 50.

    Returns:
        None
    """

    try:
        image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')]
        image_files.sort(key=natural_sort_key) 

        if not image_files:
            raise ValueError("Aucune image PNG trouvée dans le dossier spécifié.")

        images = []

        for i, image_path in enumerate(image_files):
            image = Image.open(image_path).convert("RGBA")

            draw = ImageDraw.Draw(image)

            try:
                font = ImageFont.truetype("arial.ttf", font_size)  
            except IOError:
                font = ImageFont.load_default() 

            epoch = ''.join(re.findall(r'\d+', image_path))
            draw.text((10, 10), f"Epoch {epoch}", font=font, fill=(255, 0, 0, 255))  

            images.append(image)

        images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)
        
        print(f"GIF généré avec succès : {gif_path}")
    except Exception as e:
        print(f"Une erreur est survenue lors de la création du GIF : {str(e)}")