import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np

class LSTM(nn.Module):

    def __init__(self, embedding_size = 10, hidden_size = 50, vocab_size = 16, tagset_size = 16):
        super().__init__()
        
        self.vars = nn.ParameterList()
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        # self.num_layers = num_layers
        # self.batch_size = batch_size 

        self.embedding = nn.Embedding(vocab_size, embedding_size).cuda()
        # self.lstm = nn.LSTM(embedding_size, hidden_size).cuda()
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True).cuda()
        self.dropout = nn.Dropout(0.1).cuda()
        self.fc = nn.Linear(hidden_size, tagset_size).cuda()

    def forward(self, x):
        x = x.long()
        hidden = None

        embed = self.embedding(x)
        # hidden = self._init_hidden()
        lstm_out, h = self.lstm(embed, hidden)
        # lstm_out = lstm_out[:,-1,:]
        drop_out = self.dropout(lstm_out)
        # output = self.fc(drop_out)
        output = self.fc(drop_out[:,-1,:])
        return output

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars