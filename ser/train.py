#model training code defined here
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from ser.CNN_model import Net
from ser.data import dataloaders

import typer
main = typer.Typer()

def train(params, DATA_DIR, SAVE_DIR):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    training_dataloader, validation_dataloader = dataloaders(params['batch_size'])
    
    # train
    for epoch in range(params['epochs']):
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
        # validate
        val_loss = 0
        correct = 0
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

            print(
                f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
            )

    #save shit
    # save the parameters!
    with open(f'{SAVE_DIR}/parameter.txt', 'w') as file:
        for param, info in params:
         file.write(f'{param}: {info}')
        file.write(f'Validation accuracy: {val_acc}')
        file.close()
