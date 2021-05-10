from loss import MSELoss, CrossEntropyLoss
from sgd import SGD
from linear import Linear
from sequential import Sequential
from activation import ReLU, Tanh

import matplotlib.pyplot as plt
import numpy as np

from math import pi
from torch import empty, set_grad_enabled
from tqdm import tqdm

###############################################################################

def to_one_hot(input, target):
    return target.new_zeros(
                target.size(0),
                input.size(1)).scatter(1, target.view(-1, 1), 1)

def from_one_hot(target):
    return target.max(dim=1)[1]

def generate_disc_set(nb, one_hot=False):
    input = empty(nb, 2).uniform_(0, 1)
    target = input.add(-0.5).pow(2).sum(1).sub(1/(2*pi)).sign().add(1).div(2).long()
    if not one_hot:
        target = to_one_hot(input, target)
    return input, target

###############################################################################

def compute_nb_errors(model, data_input, data_target, mini_batch_size=100, one_hot=False):
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(axis=1)
        for k in range(mini_batch_size):
            target = data_target[b + k]
            if not one_hot:
                target = target.argmax()
            if target != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors


def train_model(model, train_input, train_target, mini_batch_size=100, one_hot=False):
    creterion = MSELoss(model) if not one_hot else CrossEntropyLoss(model)
    lr = 5e-2 if not one_hot else 1e-3
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.5)
    nb_epochs = 200
    l = []

    for _ in (range(nb_epochs)):
        batch_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = creterion(output, train_target.narrow(0, b, mini_batch_size))
            
            model.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss + batch_loss
        l.append(batch_loss)
        
    return np.array(l)

###############################################################################

set_grad_enabled(False)

nb_test, nb_epochs = 10, 200
mse_test_error, cross_test_error = 0., 0.
loss_mse, loss_cross = np.zeros(nb_epochs), np.zeros(nb_epochs)

for i in tqdm(range(nb_test)):
    one_hot = False
    train_input, train_target = generate_disc_set(1000, one_hot)
    test_input, test_target = generate_disc_set(1000, one_hot)

    model = Sequential(
        Linear(2, 25),
        ReLU(),
        Linear(25, 2)
    )

    loss_mse += train_model(model, train_input, train_target, one_hot=one_hot)
    mse_test_error += compute_nb_errors(
        model, test_input, test_target, one_hot=one_hot) / len(test_input) * 100

    one_hot = True
    train_target = from_one_hot(train_target)
    test_target = from_one_hot(test_target)

    loss_cross += train_model(model, train_input, train_target, one_hot=one_hot)
    cross_test_error += compute_nb_errors(
        model, test_input, test_target, one_hot=one_hot) / len(test_input) * 100

mse_test_error /= nb_test
cross_test_error /= nb_test
loss_mse /= nb_test
loss_cross /= nb_test

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False)
ax1.plot(loss_mse, label='Test Error = {:.2f}%'.format(mse_test_error), color='red')
ax1.set_xlabel('# epochs')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()

ax2.plot(loss_cross, label='Test Error = {:.2f}%'.format(cross_test_error), color='orange')
ax2.set_xlabel('# epochs')
ax2.set_ylabel('Loss (Cross Entropy)')
ax2.legend()

plt.tight_layout()
plt.show()
