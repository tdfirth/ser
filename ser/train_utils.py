import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
import json
import pandas as pd
import visdom

def _train_batch(model_params, epoch, train_losses, vis, loss_plot):
    for batch, (images, labels) in enumerate(model_params.dataloaders['training_dataloader']):
        images, labels = images.to(model_params.device), labels.to(model_params.device)
        model_params.model.train()
        model_params.optimizer.zero_grad()
        output = model_params.model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        model_params.optimizer.step()
        print(
            f"Train Epoch: {epoch} | Batch: {batch}/{len(model_params.dataloaders['training_dataloader'])} "
            f"| Loss: {loss.item():.4f}"
        )
        train_losses.append(loss.item()) #maybe add batch and epoch info here
        vis_update(batch, epoch, loss, vis, loss_plot) 
    return

@torch.no_grad()
def _val_batch(model_params, epoch, val_accuracy, val_losses):
    val_loss = 0
    correct = 0
    val_best = 0
    for images, labels in model_params.dataloaders['validation_dataloader']:
        images, labels = images.to(model_params.device), labels.to(model_params.device)
        model_params.model.eval()
        output = model_params.model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(model_params.dataloaders['validation_dataloader'].dataset)
    val_acc = correct / len(model_params.dataloaders['validation_dataloader'].dataset)
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
    return

#plotting loss as you go
def vis_update(batch, epoch, loss, vis, loss_plot):
    vis.line(X=torch.ones((1,1)).cpu()*batch*epoch, Y =torch.ones((1,1)).cpu()*loss.item(), win = loss_plot, update='append')
    return

def save_outputs(params, model, acc_dict):
    ### SAVING THE RESULTS ###
    torch.save(model.state_dict(), params.SAVE_DIR / 'model_dict')
    
    with open(params.SAVE_DIR / 'parameters.json', 'w') as file:
        param_dict = {str(key):str(value) for key, value in dataclasses.asdict(params).items()}
        json.dump(param_dict, file, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        file.close()

    acc_dict['train_losses'].to_csv(params.RESULTS_DIR / 'train_loss.csv', index=False)
    acc_dict['val_accuracy'].to_csv(params.RESULTS_DIR / 'val_accuracy_loss.csv')

    return