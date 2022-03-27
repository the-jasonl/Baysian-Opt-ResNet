import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold


def train(dataloader, model, loss_fn, optimizer, device):
    if hasattr(dataloader.sampler, "indices"):
        size = len(dataloader.sampler.indices)
    else:
        size = len(dataloader.dataset)
    current = 0
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch += 1
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        elif batch == num_batches:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{size:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    if hasattr(dataloader.sampler, "indices"):
        size = len(dataloader.sampler.indices)
    else:
        size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def kfold_train(training_data, test_data, model, epochs, batch_size, n_splits, learning_rate, device):
    loss_fn = nn.CrossEntropyLoss()
    # set random state for reproducability
    # set seed for reproducability
    torch.manual_seed(10)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=10)
    # Create data loaders for test data
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    accuracy_scores = []
    test_losses = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(training_data)):
        print('---fold no---{}---'.format(fold+1))
        # use default model state and new optimizer for every fold
        model_copy = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model_copy.parameters(), lr=learning_rate)
        # Load subsamplers
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        cv_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
        # Create data loaders for training and cross-val data
        train_dataloader = DataLoader(
            training_data, batch_size=batch_size, sampler=train_subsampler)
        cv_dataloader = DataLoader(
            training_data, batch_size=batch_size, sampler=cv_subsampler)
        # Train fold
        for t in range(epochs):
            print(f"---Epoch {t+1}---\n")
            train(train_dataloader, model_copy, loss_fn, optimizer, device)
            test(cv_dataloader, model_copy, loss_fn, device)
        # Fold test performance
        accuracy, test_loss = test(
            test_dataloader, model_copy, loss_fn, device)
        accuracy_scores.append(accuracy)
        test_losses.append(test_loss)
    # Average metrics across folds
    avg_accuracy = sum(accuracy_scores)/len(accuracy_scores)
    avg_loss = sum(test_losses)/len(test_losses)

    return avg_accuracy, avg_loss
