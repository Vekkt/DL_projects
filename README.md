# Deep Learning Mini-projects

## https://fleuret.org/dlc/

## Project 1 – Classification, weight sharing, auxiliary losses

The objective of this project is to test different architectures to compare two digits visible in a
two-channel image. It aims at showing in particular the impact of weight sharing, and of the use of an
auxiliary loss to help the training of the main objective.

It should be implemented with PyTorch only code, in particular without using other external libraries
such as scikit-learn or numpy.

### 1 Data

The goal of this project is to implement a deep network such that, given as input a series of 2×14×14
tensor, corresponding to pairs of 14×14 grayscale images, it predicts for each pair if the first digit is
lesser or equal to the second.


The training and test set should be 1,000 pairs each, and the size of the images allows to run
experiments rapidly, even in the VM with a single core and no GPU.

You can generate the data sets to use with the function generate_pair_sets(N) defined in the file
dlc_practical_prologue.py. This function returns six tensors:

```
| Name          | Tensor dimension | Type    | Content                                  |
|:-------------:|:----------------:|:-------:|:----------------------------------------:|
| train_input   | N×2×14×14        | float32 | Images                                   |
| train_target  | N                | int64   | Class to predict ∈ { 0, 1 }              |
| train_classes | N×2              | int64   | Classes of the two digits ∈ { 0,..., 9 } |
| test_input    | N×2×14×14        | float32 | Images                                   |
| test_target   | N                | int64   | Class to predict ∈ { 0, 1 }              |
| test_classes  | N×2              | int64   | Classes of the two digits ∈ { 0,..., 9 } |
```
### 2 Objective

The goal of the project is to compare different architectures, and assess the performance improvement
that can be achieved through weight sharing, or using auxiliary losses. For the latter, the training can
in particular take advantage of the availability of the classes of the two digits in each pair, beside the
Boolean value truly of interest.

All the experiments should be done with 1,000 pairs for training and test. A convnet with ∼70'000
parameters can be trained with 25 epochs in the VM in less than 2s and should achieve ∼15% error
rate.

Performance estimates provided in your report should be estimated through 10+ rounds for each
architecture, where both data and weight initialization are randomized, and you should provide estimates
of standard deviations.

## Project 2 – Mini deep-learning framework

The objective of this project is to design a mini “deep learning framework” using only pytorch’s
tensor operations and the standard math library, hence in particular without using autograd or the
neural-network modules.

### 1 Objective

Your framework should import only torch.empty, and use no pre-existing neural-network python
toolbox. Your code should work with autograd globally off, which can be achieved with

```
torch.set_grad_enabled(False)
```
Your framework must provide the necessary tools to:

- build networks combining fully connected layers, Tanh, and ReLU,
- run the forward and backward passes,
- optimize parameters with SGD for MSE.

You must implement a test executable named test.pythat imports your framework and

- Generates a training and a test set of 1'000 points sampled uniformly in [0,1]^2 , each with a
    label 0 if outside the disk centered at (0.5, 0.5) of radius 1/√2π, and 1 inside,

- builds a network with two input units, two output units, three hidden layers of 25 units,
- trains it with MSE, logging the loss,
- computes and prints the final train and the test errors.

You should implement at least the modules Linear (fully connected layer), ReLU, Tanh, Sequential
to combine several modules in basic sequential structure, and LossMSE to compute the MSE loss.


