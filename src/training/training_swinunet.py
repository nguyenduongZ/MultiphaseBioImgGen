import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import logging
import numpy as np
import torchvision.transforms as T

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.datasets.getds import get_ds
from src.models.swinunet.encoder import SwinUnetEncoder
from src.models.swinunet.classifier import SwinUnetClassifier
from src.utils import MasterLogger, Logging, EarlyStopping, create_grid_image, setup_seed

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    OmegaConf.set_readonly(cfg, False)
    
    # MasterLogger
    logger = MasterLogger(cfg).logger
    
    # Logging Setup (WanDB + TensorBoard)
    # logging = Logging(cfg)
    
    # Seed setup
    seed = cfg.conductor["seed"]
    setup_seed(seed)

    # Device setup
    idx = cfg.conductor["trainer"]["idx"]
    torch.cuda.set_device(idx)
    device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")

    encoder = SwinUnetEncoder(
        img_size=256,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 2, 1],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        drop_rate=0.1,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUnetClassifier(encoder, num_classes=4)
    model = model.to(device)
    
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Dataset setup
    dataset_name = cfg.datasets["name"]
    train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl = get_ds(cfg=cfg, logger=logger, use_text_embedding=False)

    logger.info(f"Dataset sizes: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")
    logger.info(f"Dataloader sizes: train={len(train_dl)}, valid={len(valid_dl)}, test={len(test_dl)}")

    num_epochs = 20
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for _, (images, _, _, labels) in tqdm(enumerate(train_dl), total=len(train_dl), desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            logits, _ = model(images)
            cls_loss = criterion_cls(logits, labels)
            
            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()
            
            train_loss += cls_loss.item()

        train_loss = train_loss / len(train_dl)
        
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for _, (images, _, _, labels) in enumerate(valid_dl):
                images, labels = images.to(device), labels.to(device)

                # Forward
                logits, image_features = model(images)

                # Classification loss
                cls_loss = criterion_cls(logits, labels)

                valid_loss += cls_loss.item()

        valid_loss = valid_loss / len(valid_dl)

        scheduler.step()

        # Logging
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")

        # # Save best model
        # if valid_acc > best_acc:
        #     best_acc = valid_acc
        #     torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    main()