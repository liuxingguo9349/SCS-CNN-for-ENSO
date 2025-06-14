import torch.nn as nn

def get_loss_fn(loss_name):
    """
    Factory function to get a loss function by name.
    """
    if loss_name.lower() == 'mse':
        return nn.MSELoss()
    elif loss_name.lower() == 'l1':
        return nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
