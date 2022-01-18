import torch
import torch.nn as nn

# https://medium.com/analytics-vidhya/pytorch-for-deep-learning-binary-classification-logistic-regression-382abd97fb43
class DiscriminativeRNN(nn.Module):
    def __init__(self, data_feature_shape, num_layers=2, num_units=100,
                 scope_name="discriminatorRNN", *args, **kwargs):
        super(DiscriminativeRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=data_feature_shape[-1],
                           hidden_size=num_units,
                           num_layers=num_layers,
                           batch_first=True)
        self.classifier = nn.Linear(num_units*data_feature_shape[1], 1)

    def forward(self, x):
        # https://stackoverflow.com/questions/58092004/how-to-do-sequence-classification-with-pytorch-nn-transformer
        x, _ = self.rnn(x)
        # choose only last hidden state
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x