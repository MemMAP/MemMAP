import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from model_dclstm import dclstm
from    copy import deepcopy



class META(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):

        super(META, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
	self.bit_size = 16
        self.net = dclstm()
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def forward(self, x_spt, y_spt, x_qry, y_qry):

        task_num, setsz, c_ = x_spt.size() 
        querysz = x_qry.size(1)
        
        criterion = nn.BCEWithLogitsLoss()

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i])
            loss = criterion(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters())
                loss_q = criterion(logits_q, y_qry[i])
                losses_q[0] += loss_q

                logits_q[logits_q >= 0] = 1
                logits_q[logits_q < 0] = 0
                correct = torch.eq(logits_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights)
                loss_q = criterion(logits_q, y_qry[i])
                losses_q[1] += loss_q
                logits_q[logits_q >= 0] = 1
                logits_q[logits_q < 0] = 0     
                correct = torch.eq(logits_q, y_qry[i]).sum().item()      
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):

                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights)
                loss = criterion(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = criterion(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    logits_q[logits_q >= 0] = 1
                    logits_q[logits_q < 0] = 0     
                    correct = torch.eq(logits_q, y_qry[i]).sum().item()       
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num * self.bit_size)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry): # NOT edited yet

        querysz = x_qry.size(0)
        corrects = [0 for _ in range(self.update_step_test + 1)]
        criterion = nn.BCEWithLogitsLoss()

        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = criterion(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            logits_q = net(x_qry, net.parameters())
            logits_q[logits_q >= 0] = 1
            logits_q[logits_q < 0] = 0     
            correct = torch.eq(logits_q, y_qry).sum().item()   
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            logits_q = net(x_qry, fast_weights)
            logits_q[logits_q >= 0] = 1
            logits_q[logits_q < 0] = 0     
            correct = torch.eq(logits_q, y_qry).sum().item()   
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights)
            loss = criterion(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = criterion(logits_q, y_qry)

            with torch.no_grad():
                logits_q[logits_q >= 0] = 1
                logits_q[logits_q < 0] = 0     
                correct = torch.eq(logits_q, y_qry).sum().item()   
                corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects) / (querysz * self.bit_size)

        return accs

def main():
    pass


if __name__ == '__main__':
    main()
