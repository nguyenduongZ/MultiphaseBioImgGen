import torch

def compute_mse(original, model_output, device):
    original = original.to(device)
    model_output = model_output.to(device)
    
    # Ensure dimensions match
    if original.shape != model_output.shape:
        raise ValueError("Original and model output shapes do not match for MSE computation.")
    
    mse = torch.mean((original - model_output) ** 2).item()
    return mse
