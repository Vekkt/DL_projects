from loss import MSELoss
from sgd import SGD
from linear import Linear
from sequential import Sequential
from activation import ReLU, Tanh
from math import pi
from torch import empty, set_grad_enabled, tensor, manual_seed
import matplotlib.pyplot as plt
from tqdm import tqdm


manual_seed(0)

def pp(array):
    for p, g in array:
        print('parameter:')
        print(p)
        print('gradient:')
        print(g)
        print('--------------')


set_grad_enabled(False)

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
    loss = MSELoss(model)
    optimizer = SGD(model, lr=1e-1)
    nb_epochs = 250
    l = []
    
    stop_at, i = 2, 1

    for _ in tqdm(range(nb_epochs)):
        batch_loss = 0
        for b in range(0, train_input.size(0), 1):
            # output = model(train_input.narrow(0, b, mini_batch_size))
            # fit = loss(output, train_target.narrow(0, b, mini_batch_size))
            output = model(train_input[b])
            fit = loss(output, train_target[b])
            model.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += fit.float()
        l.append(batch_loss)
        
    plt.plot(range(nb_epochs), l)
    plt.show()


# model = Sequential(
#     Linear(2, 25, name='linear 1'),
#     ReLU('relu 1'),
#     Linear(25, 2, name='linear 2')
# )

# train_model(model, train_input, train_target)
# print(compute_nb_errors(model, test_input, test_target))

n = 2
input = train_input[:n]
target = train_target[:n]

print(input)
model = Sequential(
    Linear(2, 2),
    ReLU()
)
optimizer = SGD(model, lr=1)
loss = MSELoss(model)
# model = Linear(2, 2)
for i in range(n):
    output = model(input[i])
    fit = loss(output, target[i])
    model.zero_grad()
    pp(model.parameters())
    loss.backward()
    optimizer.step()
    print('backward pass done')
    pp(model.parameters())
    pp(model.parameters())
    print(output)
