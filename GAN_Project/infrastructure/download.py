import os
import zipfile
import requests
from .config import *

def is_data_present():

    """
    Checks if any required data files are present in the specified folder.
    """

    required_files = [
        "list_landmarks_align_celeba.csv",
        "list_attr_celeba.csv",
        "list_bbox_celeba.csv",
        "list_eval_partition.csv",
        "img_align_celeba"
    ]

    for file in required_files:
        if not os.path.exists(os.path.join(raw_data_dir, file)):
            return False

    return True

def download_and_extract_celeba():

    """
    Downloads the CelebA dataset and extracts it only if the data is not already present.
    """

    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset"


    zip_file_path = os.path.join(raw_data_dir, "celeba-dataset.zip")
    os.makedirs(raw_data_dir, exist_ok=True)


    if not is_data_present():
        print("Data is not present. Downloading...")
        response = requests.get(dataset_url, stream=True)
        if response.status_code == 200:
            with open(zip_file_path, "wb") as zip_file:
                for chunk in response.iter_content(chunk_size=1024):
                    zip_file.write(chunk)
            print("Download completed.")

            print("Extracting files...")
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(raw_data_dir)

            os.remove(zip_file_path)
            print(f"Data has been extracted to: {raw_data_dir}")
        else:
            print(f"Error during download. HTTP Status: {response.status_code}")
    else:
        print("Data is already present.")