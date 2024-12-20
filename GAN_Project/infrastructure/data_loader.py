import os
from joblib import Parallel, delayed
from .download import download_and_extract_celeba
from .transform import process_image
from .image_selector import select_images
from .config import *

def process_additional_images(n_jobs):

    """
    Processes additional images placed in the additional data folder.
    """

    

    additional_images = [img for img in os.listdir(additional_raw_data_dir) if img.endswith(('jpg', 'png', 'jpeg'))]

    print(f"Processing {len(additional_images)} additional images...")

    if n_jobs == 1:
        for image in additional_images:
            process_image(image, input_folder=additional_raw_data_dir)
    else:    
        Parallel(n_jobs=n_jobs)(delayed(process_image)(image, input_folder=additional_raw_data_dir) for image in additional_images)


def process_celeba_images(n_jobs, selected_images):
    
    """
    Processes additional images placed in the additional data folder.
    """

    selected_images_dir = os.path.join(raw_data_dir, 'img_align_celeba', 'img_align_celeba')

    print(f"Processing {len(selected_images)} additional images...")

    if n_jobs == 1:
        for image in selected_images:
            process_image(image, input_folder=selected_images_dir)
    else:    
        Parallel(n_jobs=n_jobs)(delayed(process_image)(image, input_folder=selected_images_dir) for image in selected_images)

def data_load_transform(n_image=500, n_jobs=-1):

    """
    Orchestrates the entire pipeline: downloads data, selects images, processes them.

    Args:
        n_image (int): Number of images to process.
        n_jobs (int): Number of parallel jobs (-1 for all available cores, 1 for sequential).
    """

    print("Starting pipeline execution...")
    
    # Download and extract data
    download_and_extract_celeba()
    
    # Select images
    selected_images = select_images(n_image)
    print(f"Selected {len(selected_images)} images to process.")

    # Process celeba images
    process_celeba_images(n_jobs, selected_images)

    # Process additional images
    process_additional_images(n_jobs)

    print("Pipeline successfully executed.")
