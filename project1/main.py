
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import optim
from tqdm import tqdm
from dlc_practical_prologue import *
from convNet import *

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

###############################################################################

def to_one_hot(input, target):
    size = tuple(target.size()) + (int(target.max() + 1),)
    tmp = input.new_zeros(size)
    tmp.scatter_(tmp.dim()-1, target.unsqueeze(tmp.dim()-1), 1.0)
    return tmp

###############################################################################

def train_model(model, input, target, target_classes, mini_batch_size, nb_epochs=25):
    criterion = torch.nn.MSELoss()
    criterion_aux = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    train_loss, train_accuracy = [], []

    for _ in tqdm(range(nb_epochs)):
        epoch_loss, epoch_accuracy = 0, 0
        for b in range(0, input.size(0), mini_batch_size):
            input_batch  = input.narrow(0, b, mini_batch_size)
            target_batch = target.narrow(0, b, mini_batch_size)

            prediction, classes = model(input_batch)

            loss     = criterion(prediction, target_batch.float())
            accuracy = ((prediction > 0.5) == target_batch).sum()
            
            model.zero_grad()

            if model.aux_loss:
                classes_batch = target_classes.narrow(0, b, mini_batch_size)
                aux_loss = criterion_aux(classes, classes_batch)
                loss  += aux_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss
            epoch_accuracy += accuracy * 100 / len(target)
            
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
    return train_loss, train_accuracy


def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0
    for b in range(0, input.size(0), mini_batch_size):
        prediction, _ = model(input.narrow(0, b, mini_batch_size))
        predicted_target = torch.empty(prediction.size())
        for k in range(mini_batch_size):
            predicted_target[k] = 0.0 if prediction[k] < 0.5 else 1.0
            if target[b + k] != predicted_target[k]:
                nb_errors = nb_errors + 1
    return nb_errors

###############################################################################

nb_rounds, nb_epochs, nb_models, batch_size = 20, 25, 4, 100

train_loss     = np.zeros((nb_models, nb_epochs))
train_accuracy = np.zeros((nb_models, nb_epochs))
test_error     = np.zeros(nb_models)

for round in range(nb_rounds):
    models = {
        'Vanilla'                        : PairNet(False, False),
        'Auxiliary Loss'                 : PairNet(True , False),
        'Weight Sharing'                 : PairNet(False, True),
        'Auxiliary Loss + Weight Sharing': PairNet(True , True)
    }

    data = generate_pair_sets(1000)
    train_input, train_target, train_classes = data[:3]
    test_input , test_target,  test_classes  = data[3:]

    train_classes = to_one_hot(train_input, train_classes)
    test_classes = to_one_hot(test_input, test_classes)

    print(f'{round=}')
    for i, (_, model) in enumerate(models.items()):
        loss, accuracy = train_model(
            model, 
            train_input, train_target, 
            train_classes, 
            batch_size, nb_epochs
        )
        errors = compute_nb_errors(model, test_input, test_target, 100)

        with torch.no_grad():
            train_loss    [i] = train_loss[i] + np.array(loss)
            train_accuracy[i] = train_accuracy[i] + np.array(accuracy)
            test_error    [i] = errors / len(test_input) * 100


fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(10, 5))

for i, (name, model) in enumerate(models.items()):
    model    = model
    accuracy = train_accuracy[i] / nb_rounds
    loss     = train_loss[i] / nb_rounds
    error    = test_error[i]

    print(f"\n{model.aux_loss=}, {model.weight_sharing=} | {error=:.2f}%")
    axs[0].plot(loss, label=name)
    axs[1].plot(accuracy, label=name)

axs[0].legend()
axs[1].legend()
axs[0].set_xlabel('Epochs')
axs[1].set_xlabel('Epochs')
axs[0].set_ylabel('Training Loss')
axs[1].set_ylabel('Training Accuracy (%)')

plt.savefig('convnet_results.pdf')
plt.show()
