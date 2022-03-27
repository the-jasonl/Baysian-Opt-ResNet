import torch
from torch import nn
from torch.utils.data import DataLoader


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


def run_training(training_data, test_data, model, epochs, batch_size, learning_rate, device):
    # set seed for reproducability
    torch.manual_seed(10)
    loss_fn = nn.CrossEntropyLoss()
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # 1 fold train
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    avg_accuracy, avg_loss = test(test_dataloader, model, loss_fn, device)

    return avg_accuracy, avg_loss
