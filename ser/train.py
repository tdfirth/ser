# train
import torch
import torch.nn.functional as F

from ser.models import Parameters, Data, TrainingModel


def train_model(parameters: Parameters, data: Data, training_model: TrainingModel):
    for i, (images, labels) in enumerate(data.training_dataloader):
        images, labels = images.to(training_model.device), labels.to(training_model.device)
        for epoch in range(parameters.epochs):
            training_model.model.train()
            training_model.optimizer.zero_grad()
            output = training_model.model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            training_model.optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(data.training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )
            # validate
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for images, labels in data.validation_dataloader:
                    images, labels = images.to(training_model.device), labels.to(training_model.device)
                    training_model.model.eval()
                    output = training_model.model(images)
                    val_loss += F.nll_loss(output, labels, reduction="sum").item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                val_loss /= len(data.validation_dataloader.dataset)
                val_acc = correct / len(data.validation_dataloader.dataset)

                print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")



