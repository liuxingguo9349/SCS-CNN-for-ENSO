import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ENSODataset(Dataset):
    """
    Custom PyTorch Dataset for loading ENSO data from NetCDF files.
    """
    def __init__(self, input_path, label_path, variables, lead_time, target_month):
        """
        Args:
            input_path (str): Path to the input NetCDF file.
            label_path (str): Path to the label NetCDF file.
            variables (list): List of variable names to load from input file (e.g., ['sst', 't300']).
            lead_time (int): Forecast lead time in months.
            target_month (int): The target month index (0-11).
        """
        self.input_ds = xr.open_dataset(input_path, decode_times=False)
        self.label_ds = xr.open_dataset(label_path, decode_times=False)
        self.variables = variables
        self.lead_time = lead_time
        self.target_month = target_month

        # Align time dimensions to ensure consistency
        self.num_samples = len(self.label_ds.time)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieves one sample (input tensor and label) from the dataset.
        """
        # The logic for selecting the 3-month input window is based on the original scripts.
        # It's a bit unconventional, using a fixed 36-month history in the input file.
        # `lev` in the input file seems to represent this 36-month history.
        # `lev` in the label file represents the 12 calendar months.
        
        # We predict the target_month `lead_time` months ahead.
        # The input data is a 3-month slice from a 36-month history.
        # The slice starts at `ld_mn1 = 23 - lead_time + target_month`.
        # This seems to be a specific convention from the original research.
        
        input_start_month_idx = 23 - self.lead_time + self.target_month
        input_end_month_idx = input_start_month_idx + 3
        
        # Load SST and T300 data for the 3-month window
        sst_data = self.input_ds[self.variables[0]][idx, input_start_month_idx:input_end_month_idx, :, :].values
        t300_data = self.input_ds[self.variables[1]][idx, input_start_month_idx:input_end_month_idx, :, :].values
        
        # Concatenate to form a 6-channel tensor
        input_tensor = torch.from_numpy(np.concatenate([sst_data, t300_data], axis=0)).float()
        
        # Get the corresponding label
        label = self.label_ds['pr'][idx, self.target_month, 0, 0].values
        label_tensor = torch.tensor(label, dtype=torch.float)
        
        return input_tensor, label_tensor

def get_dataloader(config, lead_time, target_month, mode='train'):
    """
    Creates a DataLoader for a specified mode (train or val).

    Args:
        config (dict): The configuration dictionary.
        lead_time (int): Forecast lead time.
        target_month (int): Target month index (0-11).
        mode (str): 'train' or 'val'.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    if mode == 'train':
        data_cfg = config['data']
        shuffle = True
        batch_size = config['training']['batch_size']
        num_workers = config['num_workers']
    elif mode == 'val':
        data_cfg = config['validation']
        shuffle = False
        # Use a larger batch size for validation for efficiency
        batch_size = config['training']['batch_size'] * 2
        num_workers = config['num_workers']
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'train' or 'val'.")

    dataset = ENSODataset(
        input_path=data_cfg['input_path'],
        label_path=data_cfg['label_path'],
        variables=data_cfg['variables'],
        lead_time=lead_time,
        target_month=target_month
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train') # Drop last incomplete batch only for training
    )
    return dataloader
