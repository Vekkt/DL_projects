from loss import MSELoss, CrossEntropyLoss
from sgd import SGD
from linear import Linear
from sequential import Sequential
from activation import ReLU, Tanh
from math import pi
from torch import empty, set_grad_enabled
import matplotlib.pyplot as plt
from tqdm import tqdm

###############################################################################

def generate_disc_set(nb, one_hot=True):
    input = empty(nb, 2).uniform_(-1, 1)
    val = input.pow(2).sum(1).sub(2 / pi).sign().add(1).div(2).long()
    if one_hot:
        target = empty(nb, 2).zero_()
        for idx, y in enumerate(target):
            y[val[idx]] = 1
    else:
        target = val
    return input, target

###############################################################################

def compute_nb_errors(model, data_input, data_target, mini_batch_size=100, one_hot=True):
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(axis=1)
        for k in range(mini_batch_size):
            target = data_target[b + k]
            if one_hot:
                target = target.argmax()
            if target != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors


def train_model(model, train_input, train_target, mini_batch_size=100, one_hot=True):
    creterion = MSELoss(model) if one_hot else CrossEntropyLoss(model)
    optimizer = SGD(model.parameters())
    nb_epochs = 1000
    l = []

    for _ in tqdm(range(nb_epochs)):
        batch_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = creterion(output, train_target.narrow(0, b, mini_batch_size))
            
            model.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss + batch_loss
        l.append(batch_loss)
        
    plt.plot(range(nb_epochs), l)

###############################################################################

set_grad_enabled(False)

one_hot = True
train_input, train_target = generate_disc_set(1000, one_hot)
test_input, test_target = generate_disc_set(1000, one_hot)

model = Sequential(
    Linear(2, 25),
    ReLU(),
    Linear(25, 2)
)

train_model(model, train_input, train_target, one_hot=one_hot)
print("error rate: {:.2f}%".format(compute_nb_errors(
    model, test_input, test_target, one_hot=one_hot) / len(test_input) * 100))
plt.show()
