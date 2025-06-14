import torch
import numpy as np
import os
from tqdm import tqdm
from ..utils.checkpoint import save_checkpoint

class Trainer:
    """
    A class to handle the training and validation loops for the model.
    """
    def __init__(self, model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, config, output_dir):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.output_dir = output_dir
        self.best_metric = -np.inf  # Initialize best metric for model saving

    def _run_one_epoch(self, epoch, is_training=True):
        """
        Runs a single epoch of either training or validation.
        """
        mode = "Training" if is_training else "Validating"
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        all_preds, all_labels = [], []
        
        data_loader = self.train_loader if is_training else self.val_loader
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']} [{mode}]")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if is_training:
                # Enable gradient calculation for saliency maps if needed
                if self.config['training'].get('calculate_saliency', False):
                    inputs.requires_grad = True

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # Saliency map calculation
                if self.config['training'].get('calculate_saliency', False) and \
                   (epoch + 1) % self.config['training'].get('saliency_interval', 1) == 0:
                    self._save_saliency_map(inputs.grad, epoch)

            else: # Validation
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)

            total_loss += loss.item()
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / len(data_loader)
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        correlation = np.corrcoef(all_preds, all_labels)[0, 1] if len(all_preds) > 1 else 0.0
        
        return avg_loss, correlation

    def _save_saliency_map(self, grad, epoch):
        """Saves the computed gradient as a saliency map."""
        saliency_dir = os.path.join(self.output_dir, "saliency_maps")
        os.makedirs(saliency_dir, exist_ok=True)
        saliency_map = grad.abs().cpu().numpy()
        np.save(os.path.join(saliency_dir, f"saliency_epoch_{epoch+1}.npy"), saliency_map)
        print(f"Saliency map saved for epoch {epoch+1}")

    def train(self):
        """
        The main training loop over all epochs.
        """
        for epoch in range(self.config['training']['epochs']):
            train_loss, train_corr = self._run_one_epoch(epoch, is_training=True)
            print(f"Epoch {epoch+1} Train | Loss: {train_loss:.4f}, Correlation: {train_corr:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")

            val_loss, val_corr = self._run_one_epoch(epoch, is_training=False)
            print(f"Epoch {epoch+1} Val   | Loss: {val_loss:.4f}, Correlation: {val_corr:.4f}")

            # Save the best model based on validation correlation
            if val_corr > self.best_metric:
                self.best_metric = val_corr
                print(f"--> New best model found with correlation: {val_corr:.4f}! Saving checkpoint...")
                save_checkpoint(
                    state={'epoch': epoch + 1, 'model_state_dict': self.model.state_dict(), 'best_metric': self.best_metric},
                    is_best=True,
                    directory=self.output_dir
                )
        
        # Save the final model state
        print("Training finished. Saving last model checkpoint.")
        save_checkpoint(
            state={'epoch': self.config['training']['epochs'], 'model_state_dict': self.model.state_dict(), 'best_metric': self.best_metric},
            is_best=False,
            directory=self.output_dir,
            filename='last_checkpoint.pth'
        )
