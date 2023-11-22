'''A modified version of https://stackoverflow.com/a/73704579 implementation for early stopping.'''

from loguru import logger
import torch

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, trace_func=print):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.trace_func = trace_func

    def early_stop(self, validation_loss):
        """
        Tracks the validation loss and returns the condition for stopping.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        return False
    
    def save_checkpoint(self, epoch, model, valid_loss, optimizer, checkpoint_path):
        """
        Saves the model checkpoint and parameters to a file in output directory
        """
        # TODO: integrate this with an earlystopping

        logger.info("Saving model parameters...")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': valid_loss,
        }

        torch.save(checkpoint, checkpoint_path)
