# train
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ser.constants import RESULTS_DIR
from ser.models import Parameters, Data, TrainingModel, BestModel
from utils import get_file_path


def train_one_batch(images: np.array, labels: np.array, training_model: TrainingModel):
    training_model.model.train()
    training_model.optimizer.zero_grad()
    output = training_model.model(images)
    loss = F.nll_loss(output, labels)
    loss.backward()
    training_model.optimizer.step()
    return loss


def get_validation_loss(data: Data, training_model: TrainingModel):
    # validate
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data.validation_dataloader:
            images, labels = images.to(training_model.device), labels.to(
                training_model.device
            )
            training_model.model.eval()
            output = training_model.model(images)
            val_loss += F.nll_loss(output, labels, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        val_loss /= len(data.validation_dataloader.dataset)
        val_acc = correct / len(data.validation_dataloader.dataset)
        return val_loss, val_acc


def train_model(parameters: Parameters, data: Data, training_model: TrainingModel):

    accuracies = []  # TODO- something with accuracy list?
    curr_best_model: Optional[BestModel] = None

    for epoch in range(parameters.epochs):
        for batch, (images, labels) in enumerate(data.training_dataloader):
            images, labels = images.to(training_model.device), labels.to(
                training_model.device
            )

            loss = train_one_batch(images, labels, training_model)

            print(
                f"Train Epoch: {epoch} | Batch: {batch}/{len(data.training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )

        val_loss, val_acc = get_validation_loss(data, training_model)
        accuracies.append((val_acc, epoch))

        if not curr_best_model or val_acc > curr_best_model.accuracy:
            curr_best_model = BestModel(
                val_acc, val_loss, epoch, training_model.model.state_dict()
            )

        print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")

    torch.save(curr_best_model.model_data, get_file_path(RESULTS_DIR, parameters.id))

    print(
        f"Best accuracy at epoch: {curr_best_model.epoch} | "
        f"Avg Loss: {curr_best_model.loss:.4f} | "
        f"Accuracy: {curr_best_model.accuracy}"
    )

    return accuracies
