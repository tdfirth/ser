import json
from dataclasses import dataclass, asdict
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ser.constants import Transform, OUTPUTS_DIR, PARAMETER_NAME
from ser.data import get_data
from utils import get_git_revision_hash


@dataclass(frozen=True)
class Parameters:
    id: str
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    git_hash: str = get_git_revision_hash()

    def __post_init__(self):
        file_path = get_file_path(self)
        file_path.open("w").write(json.dumps(asdict(self)))

    def __repr__(self):
        return (
            f"| ID: {self.id}  |"
            f"Model name: {self.name}  |"
            f"Epochs: {self.epochs}  |"
            f"Batch size: {self.batch_size}  |"
            f"Learning rate: {self.learning_rate} |"
        )


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
class BestModel:
    accuracy: float
    loss: float
    epoch: int
    model_data: torch.nn.Module


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


def get_file_path(parameters: Parameters, file_name=PARAMETER_NAME, suffix=".json"):
    path = OUTPUTS_DIR / parameters.name / parameters.id
    path.mkdir(exist_ok=True, parents=True)
    return (path / file_name).with_suffix(suffix)
