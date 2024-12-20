import os
import pandas as pd
import matplotlib.pyplot as plt
from .config import *

def get_latest_log_file():

    """
    Returns the name of the latest log file in the logs directory.

    Searches for log files with the pattern 'log_YYYYMMDD_HHMMSS.csv', 
    sorts them chronologically, and returns the name of the most recent file.

    Returns:
        str: The name of the latest log file.

    Raises:
        ValueError: If no log files matching the pattern are found.
    """

    log_files = [f for f in os.listdir(logs_dir) if f.startswith('log_') and f.endswith('.csv')]
    
    if not log_files:
        raise ValueError("Aucun fichier au format 'log_YYYYMMDD_HHMMSS.csv' trouvé dans le dossier spécifié.")

    log_files.sort()

    latest_file = log_files[-1]

    return latest_file

def monitoring(log_file=None):

    """
    Visualizes training logs and prints summary statistics.

    Plots generator and discriminator loss over epochs, and displays 
    epoch duration. If no log file is provided, the latest log file is used.

    Args:
        log_file (str, optional): Name of the log file to load. Defaults to the latest log file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified log file does not exist or is not a CSV file.
    """

    if log_file:
        logs_file_dir = os.path.join(logs_dir, log_file) 
        if not os.path.isfile(logs_file_dir) or not log_file.endswith('.csv'):
            raise FileNotFoundError(f"Le fichier de logs '{logs_file_dir}' n'existe pas ou n'est pas au bon format.")

    else:
        logs_file_dir = os.path.join(logs_dir, get_latest_log_file()) 

    df = pd.read_csv(logs_file_dir)
    
    df['Epoch'] = df['Epoch'].astype(int)

    plt.plot(df['Epoch'], df['Gen_Loss_Avg'], label='Gen Loss Avg', marker='o')
    plt.plot(df['Epoch'], df['Disc_Loss_Avg'], label='Disc Loss Avg', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.xticks(df['Epoch'])  
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(df['Epoch'], df['Epoch_Duration'], label='Epoch Duration', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Duration (seconds)')
    plt.title('Epoch Duration')
    plt.xticks(df['Epoch']) 
    plt.grid(True)
    plt.show()
    
    stats = df[['Epoch_Duration', 'Gen_Loss_Avg', 'Disc_Loss_Avg']].describe().loc[['mean', 'std', 'min', 'max']]
    print("\nStats:\n", stats)
