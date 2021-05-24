from dlc_practical_prologue import *
from convNet import *
from tqdm import tqdm
import torch
from torch import optim, nn


def to_one_hot(input, target):
    size = tuple(target.size()) + (int(target.max() + 1),)
    tmp = input.new_zeros(size)
    tmp.scatter_(tmp.dim()-1, target.unsqueeze(tmp.dim()-2), 1.0).size()
    return tmp


def train_model(model, input, target, target_classes, mini_batch_size, nb_epochs=25):
    criterion = torch.nn.MSELoss()
    criterion_aux = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in tqdm(range(nb_epochs)):
        # Train
        epoch_loss = 0
        epoch_accuracy = 0
        for b in range(0, input.size(0), mini_batch_size):
            x = input.narrow(0, b, mini_batch_size)
            if model.aux_loss:
                prediction, classes = model(x)
            else:
                prediction = model(x)
            loss = criterion(prediction, target.narrow(0, b, mini_batch_size).float())
            
            model.zero_grad()
            if model.aux_loss:
                aux_loss = criterion_aux(
                    classes, target_classes.narrow(0, b, mini_batch_size))
                loss += aux_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss


def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        x = input.narrow(0, b, mini_batch_size)
        if model.aux_loss:
            prediction, classes = model(x)
        else:
            prediction = model(x)
        predicted_target = torch.empty(prediction.size())
        for k in range(mini_batch_size):
            predicted_target[k] = 0.0 if prediction[k] < 0.5 else 1.0
            if target[b + k] != predicted_target[k]:
                nb_errors = nb_errors + 1
    return nb_errors


data = generate_pair_sets(1000)
train_input, train_target, train_classes = data[:3]
test_input, test_target, test_classes = data[3:]

train_classes = to_one_hot(train_input, train_classes)
test_classes = to_one_hot(test_input, test_classes)

model = PairNet(aux_loss=True)

train_model(model, train_input, train_target, train_classes, 100, nb_epochs=25)
errors = compute_nb_errors(model, test_input, test_target, 100)

print(test_classes[0], model(test_input)[1][0])
