import os
from datetime import datetime
from .config import *

def create_log_file():

    """
    Creates a new CSV log file in the logs directory with a timestamped filename.
    
    Returns:
        str: Path to the newly created log file.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(logs_dir, f"log_{timestamp}.csv")

    with open(log_file_path, mode='w') as file:
        file.write("Timestamp,Epoch,Epoch_Duration,Gen_Loss_Avg,Disc_Loss_Avg\n")

    return log_file_path

def log_epoch_status(log_file_path, epoch, epoch_duration, loss_avg):

    """
    Logs the status of the current epoch to the specified log file.
    
    Args:
        log_file_path (str): Path to the log file.
        epoch (int): Current epoch number.
        epoch_duration (float): Duration of the epoch in seconds.
        loss_avg (dict): Dictionary containing generator and discriminator loss values.
    """

    current_time = datetime.now()

    print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} : Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds with gen_loss={loss_avg['generator']:.4f} and disc_loss={loss_avg['discriminator']:.4f}.")

    with open(log_file_path, mode="a", encoding='UTF-8') as file:
        file.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')},{epoch + 1},{epoch_duration:.2f},{loss_avg['generator']:.4f},{loss_avg['discriminator']:.4f}\n")