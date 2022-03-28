import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from models.smallresnet import ResNet

from train_utils.training import optimize_lr


def main():
    """
    Baysian Optimization process for the learning rate of a ResNet on Fashion-MNIST
    """
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load ResNet to optimize
    model = ResNet().to(device)
    # We are only optimizing learning rate, so we fix other hyperparameters
    epochs = 5
    batch_size = 64
    n_splits = 5
    optimize_lr(training_data, test_data, model,
                epochs, batch_size, n_splits, device)

    print("Done!")


if __name__ == "__main__":
    main()
