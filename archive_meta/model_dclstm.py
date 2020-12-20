import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import  math

class dclstm(nn.Module):
    """

    """

    def __init__(self):
        super(dclstm, self).__init__()

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()

        num_embeddings = 16 #should be 2
        embedding_dim = 10 
        in_features = 50
        out_features = 16

        # Embedding layer
        weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        nn.init.normal_(weight)
        self.vars.append(weight)
        ### Add "F.embedding(input, self.weight)" to "forward"


        # LSTM layer (no weight passing)
        # lstm = nn.LSTM(10,50,2)
        # lstm._parameters.keys()
        # odict_keys(['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']) (200,10);(200,50);(200);(200)
        ##### self.lstm = nn.LSTM(embedding_dim, in_features)
        # h0 = torch.randn(1, 48, in_features)
        # self.vars.append(h0)
        # c0 = torch.randn(1, 48, in_features)
        # self.vars.append(c0)

        # LSTM layer (weight passing)
        # ref: https://stackoverflow.com/questions/50168224/does-a-clean-and-extendable-lstm-implementation-exists-in-pytorch
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

        ######    def reset_parameters(self):
        ######      std = 1.0 / math.sqrt(self.hidden_size)
        ######      for w in self.parameters():
        ######          w.data.uniform_(-std, std)
        
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
        ### Add "F.linear(input, self.weight, self.bias)" to "forward"

        # sigmoid
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, vars=None):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :return: x, loss, likelihood, kld
        """
        #vars: the fast_weight in finetune
        # x = torch.LongTensor(10000,48).random_(0,2)      ### shape: (10000,48)
        # w = vars[0]

        if vars is None:
            vars = self.vars

        # x need to be LongTensor
        w = vars[0]
        x = F.embedding(x, w)

        # 1115 problem
        # https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/13
        ### self.lstm.flatten_parameters()

        # h0, c0 = vars[1], vars[2]
        ### x, _ = self.lstm(x, None)

        outputs = []
        for x0 in torch.unbind(x, dim=1):

            # Ref: https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
            
            # x0 = x0.contiguous()
            # h0, c0 = self._init_hidden(x0)
            # h0 = h0.view(h0.size(1), -1)
            # c0 = c0.view(c0.size(1), -1)
            # x0 = x0.view(x0.size(1), -1)

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

        # x = x[:,-1,:]

        w, b = vars[5], vars[6] # vars[1], vars[2]
        x = F.linear(x, w, b)

        # Make it compatible!
        x = x[-1,-1,:,:].clone()

        # 1121 
        # x = self.sigmoid(x)

        return x

    @staticmethod
    def _init_hidden(input_):
        h = torch.zeros_like(input_.view(1, input_.size(1), -1))
        c = torch.zeros_like(input_.view(1, input_.size(1), -1))
        return h, c


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

    