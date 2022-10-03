#model training code defined here
from dataclasses import dataclass
import dataclasses
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
import pandas as pd
import visdom

from ser.CNN_model import Net
from ser.data import dataloaders
from ser.train_utils import save_outputs, _train_batch, _val_batch, vis_update

import typer
main = typer.Typer()
vis = visdom.Visdom()

@dataclass
class model_parameters():
    model: object
    optimizer: str
    device: torch.device
    dataloaders: dict
    hyperparams: dataclass

### TRAINING FUNTION ###
def train(params):
    print(f"Running experiment {params.name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    training_dataloader, validation_dataloader = dataloaders(params.DATA_DIR, params.batch_size)
    
    model_params = model_parameters(model, optimizer, device, {'training_dataloader': training_dataloader, 'validation_dataloader': validation_dataloader}, params)
    
    ### TRAINING LOOP ###
    train_losses = []
    val_accuracy = []
    val_losses = []
    loss_plot = vis.line(X = torch.zeros((1)).cpu(), Y = torch.zeros((1)).cpu(), opts=dict(showlegend=True, title='train loss', xlabel = 'batch*epoch', ylabel = 'loss', legend=['train loss']))
    
    for epoch in range(1,(params.epochs +1)):
        _train_batch(model_params, epoch, train_losses, vis, loss_plot)
        _val_batch(model_params, epoch, val_accuracy, val_losses)
    
    train_losses = pd.DataFrame({'train_loss' : train_losses})
    val_accuracy = pd.DataFrame({'val_acc':val_accuracy, 'val_loss': val_losses})
    acc_dict = {'train_losses': train_losses, 'val_accuracy' : val_accuracy}

    save_outputs(params, model, acc_dict)
   
    return 