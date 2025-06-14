import torch
import argparse
import os
import yaml
import random
import numpy as np

from src.utils.config_parser import load_config, override_config_with_args
from src.data_loader import get_dataloader
from src.models.scs_cnn import SCS_CNN
from src.training.trainer import Trainer
from src.training.losses import get_loss_fn
from src.training.lr_scheduler import get_scheduler

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    """Main function to orchestrate the training and validation process."""
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Override config with command-line arguments
    config = override_config_with_args(config, args)

    # Set seed for the specific ensemble member
    set_seed(args.ens)

    # Create output directory
    run_name = config['run_name'].format(lead_time=args.lead, target_month=args.target, ens=args.ens)
    output_dir = os.path.join(config['output_base_dir'], run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the final configuration to the output directory
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get DataLoaders
    print("Loading datasets...")
    train_loader = get_dataloader(config, args.lead, args.target - 1, mode='train') # Months are 0-indexed
    val_loader = get_dataloader(config, args.lead, args.target - 1, mode='val')

    # Initialize Model
    print(f"Initializing model: {config['model']['name']}")
    model = SCS_CNN(**config['model']['params']).to(device)

    # Load pre-trained checkpoint if specified
    if 'pretrain_checkpoint_path' in config and config['pretrain_checkpoint_path']:
        checkpoint_path = config['pretrain_checkpoint_path']
        if os.path.exists(checkpoint_path):
            print(f"Loading pre-trained weights from: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
        else:
            print(f"Warning: Pre-trained checkpoint not found at {checkpoint_path}. Starting from scratch.")

    # Setup Optimizer, Scheduler, Loss
    optimizer = torch.optim.SGD(model.parameters(), **config['training']['optimizer']['args'])
    scheduler = get_scheduler(optimizer, config['training']['scheduler'], len(train_loader), config['training']['epochs'])
    loss_fn = get_loss_fn(config['training']['loss_fn'])

    # Initialize and run Trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        output_dir=output_dir
    )
    
    print(f"Starting training for run: {run_name}")
    trainer.train()
    print(f"Training finished for run: {run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SCS-CNN for ENSO Forecasting")
    parser.add_argument('--config', type=str, required=True, help="Path to the base configuration YAML file.")
    parser.add_argument('--lead', type=int, required=True, help="Lead time in months.")
    parser.add_argument('--target', type=int, required=True, help="Target month (1-12).")
    parser.add_argument('--ens', type=int, default=1, help="Ensemble member number for seeding.")
    
    # Optional arguments to override config file
    parser.add_argument('--device', type=str, help="Override device (e.g., 'cuda:1').")
    parser.add_argument('--pretrain_checkpoint_path', type=str, help="Path to pre-trained model checkpoint.")
    
    args = parser.parse_args()
    main(args)
