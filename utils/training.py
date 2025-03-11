import os, sys
import matplotlib.pyplot as plt

from .build_opt import Opt

class EarlyStopping:
    def __init__(self, opt: Opt):
        self.opt = opt
        self.patience = opt.conductor["early_stopping"]["patience"]
        self.min_delta = opt.conductor["early_stopping"]["min_delta"]
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, valid_loss):
        if valid_loss < self.best_loss - self.min_delta:
            self.best_loss = valid_loss
            self.counter = 0
            
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.opt.logger.debug(f"Early stopping activated after {self.counter} iterations!")
            return True
        
        return False
    
def plot_learning_curve(train_losses, valid_losses, fold):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(valid_losses, label='Validation Loss', color='red')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Learning Curve for Fold {fold}")
    plt.legend()
    plt.grid()
    
    save_path = f"learning_curve_fold_{fold}.png"
    plt.savefig(save_path)
    print(f"Learning Curve saved at {save_path}")
    plt.show()