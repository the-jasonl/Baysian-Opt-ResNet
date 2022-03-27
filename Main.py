import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from models.smallresnet import ResNet

from train_utils.training import run_training


def main():
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

    epochs = 5
    batch_size = 64
    learning_rate = 0.01
    run_training(training_data, test_data, model, epochs,
                 batch_size, learning_rate, device)

    print("Done!")


if __name__ == "__main__":
    main()
