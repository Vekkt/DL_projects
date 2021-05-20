from dlc_practical_prologue import *
from convNet import *


data = generate_pair_sets(100)
train_input, train_target, train_classes = data[:3]
test_input, test_target, test_classes = data[3:]

model = PairNet(aux_loss=True)

