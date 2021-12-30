import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from discriminative_score.model import DiscriminatorRNN
from data import DCData
import numpy as np

# choose dataset and model
dataset_name = 'index_growth_1mo'
gan_type = 'RNN'
run = '1'
epoch = '380'
gen_flag = False
# load data
real_data_path = "data/{}/data_feature_n_g.npy".format(dataset_name)
fake_data_path = "runs/{}/{}/{}/checkpoint/epoch_{}/generated_samples.npz".format(dataset_name, gan_type, run, epoch)
real_feature = np.load(real_data_path)[:, :, :-2]
fake_feature = np.load(fake_data_path)
fake_feature = fake_feature['sampled_features']
# initiate dataset
dataset = DCData(real_feature=real_feature, fake_feature=fake_feature)
# train test split
batch_size = 20
train_size = 0.8
train_len = int(len(dataset)*0.8)
test_len = int(len(dataset)*(1-train_size))
if (train_len + test_len) != len(dataset):
    train_len += 1
lengths = [train_len, test_len]
train_set, test_set = random_split(dataset, lengths)
train_dl = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
# hyper parameters
learning_rate = 0.00001
epochs = 700
# Model , Optimizer, Loss
model = DiscriminatorRNN(input_size=dataset.features.shape[-1])
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
loss_fn = nn.BCELoss()

# forward loop
losses = []
accur = []
model.train()
for i in range(epochs):
    for j, (x_train, y_train) in enumerate(train_dl):
        # calculate output
        output = model(x_train)

        # calculate loss
        loss = loss_fn(output, y_train.reshape(-1, 1))

        # accuracy
        # predicted = model(torch.tensor(x, dtype=torch.float32))
        output = output.reshape(-1).detach().numpy().round()
        acc = (output == y_train.detach().numpy()).mean()  # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i % 1 == 0:
        losses.append(loss)
        # accur.append(acc)
        #print("epoch {}\tloss : {}".format(i, loss))
        print("epoch {}\tloss : {}\t accuracy : {}".format(i, loss, acc))