from loss import MSELoss, CrossEntropyLoss
from sgd import SGD
from linear import Linear
from sequential import Sequential
from activation import ReLU, Tanh

from math import pi
from torch import empty, set_grad_enabled

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
    criterion = MSELoss(model) if not one_hot else CrossEntropyLoss(model)
    lr = 5e-3 if not one_hot else 5e-3
    mm = 0.9 if not one_hot else 0.6
    optimizer = SGD(model.parameters(), lr=lr, momentum=mm)
    nb_epochs = 250

    for e in range(nb_epochs):
        epoch_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(
                output, train_target.narrow(0, b, mini_batch_size))
            
            model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = loss + epoch_loss

        print("{} for epoch {}: {:.2f}".format(criterion, e, epoch_loss))
    print('-'*30)

###############################################################################

set_grad_enabled(False)

one_hot = False
train_input, train_target = generate_disc_set(1000, one_hot)
test_input, test_target = generate_disc_set(1000, one_hot)

model = Sequential(
    Linear(2, 25),
    ReLU(),
    Linear(25, 2)
)

loss_mse = train_model(model, train_input, train_target, one_hot=one_hot)
mse_train_error = compute_nb_errors(model, train_input, train_target, one_hot=one_hot)
mse_test_error = compute_nb_errors(model, test_input, test_target, one_hot=one_hot)

print("Final train error: {:.2f}%".format(100 * mse_train_error / len(train_target)))
print("Final test error: {:.2f}%".format(100 * mse_test_error / len(test_target)))
