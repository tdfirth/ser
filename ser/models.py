from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ser.constants import Transform
from ser.data import get_data


@dataclass(frozen=True)
class Parameters:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float



@dataclass()
class TrainingModel:
    device: torch.device
    _model: nn.Module
    _optimizer: Callable
    parameters: Parameters

    @property
    def model(self):
        return self._model.to(self.device)

    @property
    def optimizer(self):
        return self._optimizer(
            self.model.parameters(), lr=self.parameters.learning_rate
        )


@dataclass(frozen=True)
class Data:
    training_dataloader: DataLoader
    validation_dataloader: DataLoader

    @classmethod
    def from_inputs(cls, transform: Transform, parameters: Parameters):
        train = get_data(transform=transform, batch_size=parameters.batch_size)
        validate = get_data(
            transform=transform,
            train=False,
            shuffle=False,
            batch_size=parameters.batch_size,
        )

        return cls(train, validate)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
