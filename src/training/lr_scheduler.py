import torch
import math

class CosineAnnealingWithWarmup:
    """
    Custom learning rate scheduler that combines a linear warmup phase with
    a cosine annealing phase.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.current_epoch <= self.warmup_epochs:
                # Linear warmup
                lr = base_lr * (self.current_epoch / self.warmup_epochs)
            else:
                # Cosine annealing
                progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def get_scheduler(optimizer, scheduler_config, steps_per_epoch, total_epochs):
    """
    Factory function to get a learning rate scheduler.
    This version is simplified as the original code already provided the logic.
    """
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == 'cosine_warmup':
        # The logic in the original code was per-step, let's adapt it.
        # total_steps = total_epochs * steps_per_epoch
        # warmup_steps = scheduler_config['args']['warmup_epochs'] * steps_per_epoch
        
        # The function from the original code for LambdaLR
        def lr_lambda(current_step):
            num_step = steps_per_epoch
            epochs = total_epochs
            warmup_epochs = scheduler_config['args']['warmup_epochs']
            warmup_factor = 1e-3 # As per original code
            end_factor = 1e-6 # As per original code

            if current_step < warmup_epochs * num_step:
                alpha = float(current_step) / (warmup_epochs * num_step)
                return warmup_factor * (1 - alpha) + alpha
            else:
                current_step_after_warmup = current_step - warmup_epochs * num_step
                cosine_steps = (epochs - warmup_epochs) * num_step
                cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step_after_warmup / cosine_steps))
                return (1 - end_factor) * cosine_decay + end_factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_config['args'])
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
