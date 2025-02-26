import torch

class EarlyStopping:
    def __init__(self, patience=1000, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False
        
    def __call__(self, valid_loss):
        if valid_loss < self.best_loss - self.min_delta:
            self.best_loss = valid_loss
            self.counter = 0
            
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.stop = True
                
        return self.stop