import torch
import torch.nn.functional as F
import time

import json

def trainer(epochs,training_dataloader, validation_dataloader,device,model,optimizer,name):
    
    best_epoch = ""
    best_validation = 0
    
    for epoch in range(epochs):
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

            if best_validation < val_acc:
                best_epoch = epoch
                best_validation = val_acc

                # records model
                model_scripted = torch.jit.script(model) # Export to TorchScript
                # date and time
                timestr = time.strftime("%Y%m%d-%H%M%S")
                model_scripted.save(f"{name}_dmy{timestr}_t.pt")
                
    # records validation accuracy
    with open(f'{name}_dmy{timestr}_validation.json', "a") as f:
        json.dump({"experiment_name": name, "datetime": timestr,
        "last_validation_accuracy": val_acc,
        "best_epoch": best_epoch,
        "best_validation": best_validation}, f)
