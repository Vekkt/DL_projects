from loss import MSELoss
from sgd import SGD
from linear import Linear
from sequential import Sequential
from activation import ReLU
from math import pi
from torch import empty, set_grad_enabled, tensor
import matplotlib.pyplot as plt


def generate_disc_set(nb):
    input = empty(nb, 2).uniform_(-1, 1)
    target = empty(nb, 2).zero_()
    val = input.pow(2).sum(1).sub(2 / pi).sign().add(1).div(2).long()
    for idx, y in enumerate(target):
        y[val[idx]] = 1
    return input, target


train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

mini_batch_size = 100

set_grad_enabled(False)

def CrossEntropyLoss(input, target):
    return -input[:, target] + input.exp().sum(axis=1).log()


def compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(axis=1)
        for k in range(mini_batch_size):
            if data_target[b + k].argmax() != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


def train_model(model, train_input, train_target):
    criterion = MSELoss(model)
    optimizer = SGD(model, lr=1e-1)
    nb_epochs = 250
    l = []

    for e in range(nb_epochs):
        batch_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            
            model.zero_grad()
            criterion.backward()
            optimizer.step()

            batch_loss += loss.float()
        l.append(batch_loss)
    plt.plot(range(nb_epochs), l)
    plt.show()

set_grad_enabled(False)

model = Sequential(
    Linear(2, 128, name='linear 1'),
    ReLU('relu 1'),
    Linear(128, 2, name='linear 2')
)

train_model(model, train_input, train_target)
print(compute_nb_errors(model, train_input, train_target))
