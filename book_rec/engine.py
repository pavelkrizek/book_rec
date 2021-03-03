import torch
import torch.nn as nn


def train(data_loader, model, optimizer):
    """
    Train the model with optimizer with data from data_loader
    """
    model.train()
    for data in data_loader:
        features, targets = data
        optimizer.zero_grad()
        predictions = model(features)
        loss = nn.MSELoss()(predictions, targets.view(-1, 1))
        loss.backward()
        optimizer.step()


def evaluate(data_loader, model):
    """
    Evaluate model based on data from data_loader
    """
    final_predictions = []
    final_targets = []
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            features, targets = data
            predictions = model(features)
            predictions = predictions.numpy().tolist()
            targets = targets.numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)
    return final_predictions, final_targets
