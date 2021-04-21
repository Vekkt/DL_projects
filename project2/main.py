from loss import MSELoss
from sgd import SGD
from linear import Linear
from sequential import Sequential
from activation import ReLU, Tanh
from math import pi
from torch import empty, set_grad_enabled
import matplotlib.pyplot as plt
from tqdm import tqdm

set_grad_enabled(False)

def generate_disc_set(nb):
    input = empty(nb, 2).uniform_(-1, 1)
    target = empty(nb, 2).zero_()
    val = input.pow(2).sum(1).sub(2 / pi).sign().add(1).div(2).long()
    for idx, y in enumerate(target):
        y[val[idx]] = 1
    return input, target


train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(10000)

set_grad_enabled(False)

def CrossEntropyLoss(input, target):
    return input[:, target].mul(-1).add(input.exp().sum(axis=1).log())


def compute_nb_errors(model, data_input, data_target, mini_batch_size=100):
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(axis=1)
        for k in range(mini_batch_size):
            if data_target[b + k].argmax() != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


def train_model(model, train_input, train_target, mini_batch_size=50):
    creterion = MSELoss(model)
    optimizer = SGD(model.parameters())
    nb_epochs = 500
    l = []
    
    stop_at, i = 2, 1

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

model = Sequential(
    Linear(2, 25, name='linear 1'),
    ReLU('relu 1'),
    Linear(25, 2, name='linear 2')
)

train_model(model, train_input, train_target)
print("error rate: {:.2f}%".format(compute_nb_errors(model, test_input, test_target) / len(test_input) * 100))
plt.show()
