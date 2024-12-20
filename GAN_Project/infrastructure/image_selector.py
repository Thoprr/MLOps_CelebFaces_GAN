import os
import pandas as pd
import numpy as np
from .config import *

def select_images(n_images=20000):
    
    """
    Selects and returns a list of images to be processed based on landmark data.
    """

    df = pd.read_csv(os.path.join(raw_data_dir, "list_landmarks_align_celeba.csv"), sep=",")
    df.set_index("image_id", inplace=True)

    df_normalized = (df - df.mean(axis=0)) / df.std(axis=0)
    df_abs_normalized = np.abs(df_normalized)
    df_abs_normalized_sum = df_abs_normalized.sum(axis=1)

    selected_images = df_abs_normalized_sum.sort_values()[:n_images].index
    return selected_images.sort_values()