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

## Short Intro to Baysian Optimization
Baysian Optimization (BO) is an approach to optimize model hyperparameters. BO uses prior samples to build a function (surrogate function) that approximates how a hyperparameter affects the model performance. It then selects the most promising hyperparameter value based on an acquisition function. This process can be done iteratively, by evaluating the next sample, updating the prior and the choosing the next best parameter. BO is useful when the target function to approximate is expensive, as it provides an optimized sampling strategy.

## Model and training details

I will use the readily available Fashion-MNIST dataset for this show case and a small ResNet. The ResNet has been kept very small to streamline the training time. Similarly, hyperparameters other than training time have been optimized with regards to training time.

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

The model training process uses a k-fold cross-validation. The averaged validation score across the folds after the training process is used for the next BO step.

Since we are only optimizing learning rate, other training hyperparameters are fixed at:
```python
epochs = 5
batch_size = 64
n_splits = 5
```

## Optimization process

A Gaussian Process (GP) is used for the BO process to get a function distribution rather than a single function estimate. Moreover, a function distribution incorporates uncertainty, which is helpful for the GP to sample the next point (allowing it to consider exploration and exploitation) (see iteration 2, where the surrogate function is linear with only 1 prior sample and does not provide a clear indication for the next sample. The expected improvement is highest furthest away from the already explored point)

`sklearn` provides a `GaussianProcessRegressor` and implementation of several Kernels for the Gaussian Process. For this process, I use the widely used `Matern` Kernel with a noise level that has been set to reflect the average deviation from the mean in the k-fold scores during a sample run.

To get the next sampling location, I use expected improvement as the acquisition function.


## Sample output

Each iteration of the optimization is plotted starting from iteration 2 onwards (after 1 point has been generated). The surrogate function with `1.645` uncertainty is plotted on the left side and the acquisition function on the right side.

<img alt="optimization_plots" src="https://github.com/the-jasonl/Baysian-Opt-ResNet/blob/main/optimization_plots.png?raw=true"><br>

The optimal learning rate is printed to the shell:
```bash
Optimal learning rate 0.1328678827243221
```

In case of questions, feel free to open an issue!
