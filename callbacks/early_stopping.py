'''A modified version of https://stackoverflow.com/a/73704579 implementation for early stopping.'''

from loguru import logger
import torch

class EarlyStopper:
    """
    Monitors validation loss and performs early stopping if no improvement is observed.
    """
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, logs_path=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.logs_path = logs_path

        logger.add(self.logs_path, rotation="10 MB", level="INFO")

    def __call__(self, validation_loss, epoch, model, optimizer):
        score = -validation_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(validation_loss, model, optimizer, epoch)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(validation_loss, model, optimizer, epoch)
            self.counter = 0
    
    def save_checkpoint(self, valid_loss, model, optimizer, epoch):
        self.trace_func(f"Valid loss improved from {self.val_loss_min:.6f} to {valid_loss:.6f}. Saving model ...")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': valid_loss,
            'best_score': self.best_score,
            'counter': self.counter
        }

        torch.save(checkpoint, self.path)

        self.val_loss_min = valid_loss
