import torch
import torch.nn as nn

# https://medium.com/analytics-vidhya/pytorch-for-deep-learning-binary-classification-logistic-regression-382abd97fb43
class DiscriminatorRNN(nn.Module):
    def __init__(self, input_size, num_layers=2, num_units=100,
                 scope_name="discriminatorRNN", *args, **kwargs):
        super(DiscriminatorRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=num_units,
                           num_layers=num_layers,
                           batch_first=True)
        self.classifier = nn.Linear(num_units, 1)

    def forward(self, x):
        # https://stackoverflow.com/questions/58092004/how-to-do-sequence-classification-with-pytorch-nn-transformer
        x, _ = self.rnn(x)
        # choose only last hidden state
        x = x[:, -1, :]
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x