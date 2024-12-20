import sys
import os
from PIL import Image
from rembg import remove
from .config import *

def process_image(file, input_folder):

    """
    Processes a single image by removing the background and resizing it.
    """

    image_shape = tuple(image_size)[::-1]

    input_path = os.path.join(input_folder, file)
    output_path = os.path.join(processed_data_dir, file[:-3] + "png")

    image = Image.open(input_path)
    image = image.convert("RGBA")

    output_image = remove(image, bgcolor=(54, 99, 4, 255))
    output_image = output_image.resize(image_shape, Image.LANCZOS)

    output_image.save(output_path)
    print(f"Processed image: {file}" , flush=True)
    sys.stdout.flush()