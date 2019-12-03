import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import  math

class dclstm(nn.Module):

    def __init__(self):
        super(dclstm, self).__init__()

        self.vars = nn.ParameterList()

        num_embeddings = 16
        embedding_dim = 10 
        in_features = 50
        out_features = 16

        # Embedding layer
        weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        nn.init.normal_(weight)
        self.vars.append(weight)

        # LSTM layer
        i2h_weight = nn.Parameter(torch.Tensor(4*in_features, embedding_dim))
        i2h_bias = nn.Parameter(torch.Tensor(4*in_features))
        nn.init.kaiming_uniform_(i2h_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(i2h_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(i2h_bias, -bound, bound)

        h2h_weight = nn.Parameter(torch.Tensor(4*in_features, in_features))
        h2h_bias = nn.Parameter(torch.Tensor(4*in_features))
        nn.init.kaiming_uniform_(h2h_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(h2h_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(h2h_bias, -bound, bound)

        self.vars.append(i2h_weight)
        self.vars.append(i2h_bias)
        self.vars.append(h2h_weight)
        self.vars.append(h2h_bias)
        
        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # Fully-connected layer
        weight = nn.Parameter(torch.Tensor(out_features, in_features))
        bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(weight)
        self.vars.append(bias)

    def forward(self, x, vars=None):

        if vars is None:
            vars = self.vars

        w = vars[0]
        x = F.embedding(x, w)
        outputs = []

        for x0 in torch.unbind(x, dim=1):

            # initialization
            h0 = torch.cuda.FloatTensor(x0.shape[0],50).random_(0,1)
            c0 = torch.cuda.FloatTensor(x0.shape[0],50).random_(0,1)

            w1, b1, w2, b2 = vars[1], vars[2], vars[3], vars[4]
            F1 = F.linear(x0, w1, b1)
            F2 = F.linear(h0, w2, b2)
            preact = F1+F2
            gates = preact[:, :3 * 50].sigmoid()
            g_t = preact[:, 3 * 50:].tanh()
            i_t = gates[:, :50]
            f_t = gates[:, 50:2 * 50]
            o_t = gates[:, -50:]
            c_t = torch.mul(c0, f_t) + torch.mul(i_t, g_t)
            h_t = torch.mul(o_t, c_t.tanh())
            h_t = h_t.view(1, h_t.size(0), -1)
            c_t = c_t.view(1, c_t.size(0), -1)
            outputs.append(h_t.clone())

        x = torch.stack(outputs, dim=1)
        x = self.dropout(x)
        w, b = vars[5], vars[6] # vars[1], vars[2]
        x = F.linear(x, w, b)
        x = x[-1,-1,:,:].clone()

        return x

    def zero_grad(self, vars=None):

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

        return self.vars

    