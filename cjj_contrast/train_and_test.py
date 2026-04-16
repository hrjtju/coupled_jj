from tqdm import tqdm

import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn



def train(model, train_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        
        # perform contrastive learning 
        pred = model(x)
        
        # compute loss
        
    
@torch.no_grad()
def test(model, test_loader):
    ...