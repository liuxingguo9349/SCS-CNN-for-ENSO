import torch
import os
import shutil

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth'):
    """
    Saves a model checkpoint.

    Args:
        state (dict): Contains model's state_dict and other info.
        is_best (bool): If true, copies the checkpoint to 'best_model.pth'.
        directory (str): Directory to save the checkpoint.
        filename (str): The name of the checkpoint file.
    """
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(directory, 'best_model.pth'))
        print(f"Best model saved to {os.path.join(directory, 'best_model.pth')}")
