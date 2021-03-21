from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size): # input_size = number of tokens in Japanese, hidden_size=rand number
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size=hidden_size, num_layers=2, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        #self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=False)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, (hidden_state, cell_state) = self.lstm(embedded)
        return hidden_state, cell_state


class DecoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size): # hidden_size=rand number (same that EncoderDNN() used, output_size = total tokens of English
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers=2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden_state, cell_state):
        x = input.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden_state, cell_state) = self.LSTM(embedding, (hidden_state, cell_state))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden_state, cell_state