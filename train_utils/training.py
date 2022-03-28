import copy
from typing import Tuple
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from gaussian_utils.gaussian import GaussianProcessOptimizer
from plotting_utils.plotting import plot_approximation, plot_acquisition


def train(dataloader: DataLoader, model: torch.nn.Module, loss_fn: nn.CrossEntropyLoss,
          optimizer: torch.optim.SGD, device: str) -> None:
    """Train pipeline for model

    Args:
        dataloader (DataLoader): Torch Dataloader for training data
        model (torch.nn.Module): model (torch.nn.Module): Model to be trained
        loss_fn (nn.CrossEntropyLoss): loss function
        optimizer (torch.optim.SGD): optimizer
        device (str): "cpu" or "cuda"
    """
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


def test(dataloader: DataLoader, model: torch.nn.Module,
         loss_fn: nn.CrossEntropyLoss, device: str) -> Tuple[float, float]:
    """Return performance of model on a test set

    Args:
        dataloader (DataLoader): Torch Dataloader for test data
        model (torch.nn.Module): Model to be evaluated
        loss_fn (nn.CrossEntropyLoss): loss function
        device (str): "cpu" or "cuda"

    Returns:
        Tuple[float, float]: return test accuracy and loss
    """
    if hasattr(dataloader.sampler, "indices"):
        # Use subsample indices if subsampler is used
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


def kfold_train(training_data: torch.Tensor, test_data: torch.Tensor,
                model: torch.nn.Module, epochs: int, batch_size: int, n_splits: int,
                learning_rate: float, device: str) -> Tuple[float, float]:
    """Trains n_splits models on n_splits folds and returns the average accuracy and loss

    Args:
        training_data (torch.Tensor): Data to be trained on
        test_data (torch.Tensor): Data for testing
        model (torch.nn.Module): Model to be trained and tested
        epochs (int): Epochs to train model for
        batch_size (int): Training batch size
        n_splits (int): Number of folds for training
        learning_rate (float): Model learning rate
        device (str): "cpu" or "cuda"

    Returns:
        Tuple[float, float]: average accuracy and loss across folds
    """
    loss_fn = nn.CrossEntropyLoss()
    # set random state for reproducibility
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


def optimize_lr(
        training_data: torch.Tensor, test_data: torch.Tensor,
        model: torch.nn.Module, epochs: int, batch_size: int,
        n_splits: int, device: str) -> None:
    """Optimizes the learning rate of the given model with provided data and hyperparameters. 
    Creates plots during optimizations, saving them to file optimization_plots.png

    Args:
        training_data (torch.Tensor): Data to be trained on
        test_data (torch.Tensor): Data for testing
        model (torch.nn.Module): Model to be trained and tested
        epochs (int): Epochs to train model for
        batch_size (int): Training batch size
        n_splits (int): Number of folds for training
        device (str): "cpu" or "cuda"
    """
    lb = 0.001  # lower bound for param to be optimized
    ub = 0.9    # upper bound for param to be optimized
    max_evals = 10  # num of params to try
    n_restarts = 10  # restarts of gpo
    gpo = GaussianProcessOptimizer(lb, ub, n_restarts)

    plt.figure(figsize=(12, max_evals * 3))
    plt.subplots_adjust(hspace=0.4)

    for i in range(max_evals):
        print("---Param no----{}".format(i+1))
        # Get next set of parameters to try
        if i < 1:
            learning_rate = 0.01   # start with a lr thats often good
        else:
            gpo.gp = gpo.fit()
            learning_rate = gpo.next_point()
            # plotting from second iteration onwards
            plt.subplot(max_evals, 2, 2 * (i-1) + 1)
            plot_approximation(gpo.gp, gpo.X, gpo.X_samples, gpo.Y_samples,
                               X_next=learning_rate, show_legend=(i == 1))
            plt.title(f'Iteration {i+1}')
            plt.subplot(max_evals, 2, 2 * (i-1) + 2)
            plot_acquisition(gpo.X, gpo.expected_improvement(
                gpo.X, gpo.X_samples, gpo.gp), X_next=learning_rate, show_legend=(i == 1))
        print("Learning rate:", learning_rate)
        avg_accuracy, avg_loss = kfold_train(
            training_data, test_data, model, epochs, batch_size, n_splits, learning_rate, device)
        # Store parameters and scores
        gpo.add_point(learning_rate, avg_loss)

    optimal_lr = gpo.best_point()
    print("Optimal learning rate", optimal_lr)
    # Save plot to file
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    plt.savefig(timestamp + '_optimization_plots.png')
