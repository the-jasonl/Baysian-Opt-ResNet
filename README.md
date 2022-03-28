# Baysian Optimization process for the learning rate of a ResNet on Fashion-MNIST

This process showcases a Baysian Optimization process for the learning rate of a small ResNet adaptation on the Fashion-MNIST dataset.

## Installation

Use the package manager [pip3](https://pip.pypa.io/en/stable/) to setup the environment.

```bash
pip3 install --no-cache-dir -r requirements.txt
```

The repo has been created with `Python 3.9.7` and has been tested on MacOS and Ubuntu.

## Usage

```bash
python3 Main.py
```

The script prints the result out to console and saves a plot of the optimization proceess.

## Model and training details

This process serves as a showcase, so the ResNet has been kept very small to streamline the training time. Similarly, other hyperparameters have been kept small with regards to training time.

The showcased ResNet consists of 2 ResBlocks:

```python
ResNet(
  (conv1): Conv2d(1, 8, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
  (bn1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU()
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (resblock1): ResBlock(
    (conv1): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)
    (bn1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)
    (bn2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU()
  )
  (resblock2): ResBlock(
    (downsample): Sequential(
      (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), bias=False)
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (conv1): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2))
    (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
    (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU()
  )
  (fc): Linear(in_features=576, out_features=10, bias=True)
)
```


Since we are only optimizing learning rate, other hyperparameters are fixed at:
```python
epochs = 5
batch_size = 64
n_splits = 5
```

## Sample output

```bash
Optimal learning rate 0.1328678827243221
```

<img alt="optimization_plots" src="https://github.com/the-jasonl/Baysian-Opt-ResNet/blob/main/optimization_plots.png?raw=true"><br>


In case of questions, feel free to open an issue!
