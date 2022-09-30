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

from ser.CNN_model import Net
from ser.data import dataloaders

import typer
main = typer.Typer()

### TRAINING FUNTION ###
def train(params, DATA_DIR, SAVE_DIR, RESULTS_DIR, commit):
    print(f"Running experiment {params.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    training_dataloader, validation_dataloader = dataloaders(DATA_DIR, params.batch_size)
    
    # train
    train_losses = []
    val_accuracy = []
    val_losses = []
    for epoch in range(params.epochs):
        for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )
            train_losses.append(loss.item()) #maybe add batch and epoch info here
        
        # validate
        val_loss = 0
        correct = 0
        val_best = 0
        with torch.no_grad():
            for images, labels in validation_dataloader:
                images, labels = images.to(device), labels.to(device)
                model.eval()
                output = model(images)
                val_loss += F.nll_loss(output, labels, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            val_loss /= len(validation_dataloader.dataset)
            val_acc = correct / len(validation_dataloader.dataset)
            if val_acc > val_best:
                val_best = val_acc
                best_epoch = epoch
            print(
                f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
            )
            val_accuracy.append(val_acc)
            val_losses.append(val_loss)
    print(
        f"|* Best Epoch: {best_epoch} | Best Accuracy: {val_best} *|"
        ) 

    ### SAVING THE RESULTS ###
    torch.save(model.state_dict(), SAVE_DIR / 'model_dict')
    
    with open(SAVE_DIR / 'parameters.json', 'w') as file:
        json.dump(dataclasses.asdict(params), file, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        file.close()

    train_losses = pd.DataFrame({'train_loss' : train_losses})
    val_accuracy = pd.DataFrame({'val_acc':val_accuracy, 'val_loss': val_losses})
    train_losses.to_csv(RESULTS_DIR / 'train_loss.csv', index=False)
    val_accuracy.to_csv(RESULTS_DIR / 'val_accuracy_loss.csv')

    return 
