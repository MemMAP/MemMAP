import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from model_dclstm import dclstm
#from    LSTM import LSTM
from    copy import deepcopy



class META(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """

        :param args:
        """
        super(META, self).__init__()

        self.update_lr = args.update_lr #beta
        self.meta_lr = args.meta_lr #alpha
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = dclstm()
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        print("init")


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm = total_norm + param_norm.item() ** 2
            counter = counter + 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_ = x_spt.size() 
        querysz = x_qry.size(1)
        
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.BCELoss()

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        # Change from Long to Float
        y_spt = y_spt.clone().type(torch.cuda.FloatTensor)
        y_qry = y_qry.clone().type(torch.cuda.FloatTensor)
        
        for i in range(task_num):

            ##### print(i, '-th task started')
            
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i])
            # pred = self.net(x_spt[i])
            
            # Ta-Yang 1107
            # y_spt[i] = y_spt[i].long()
            # logits = logits.long()
            
            loss = criterion(logits, y_spt[i])
            # loss = criterion(pred, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters()))) # ???
            #print("maml-103")

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters())
                # pred_q = self.net(x_qry[i], self.net.parameters())
                loss_q = criterion(logits_q, y_qry[i])
                # loss_q = criterion(pred_q, y_qry[i])
                losses_q[0] += loss_q
                
               # pred_q = F.sigmoid(logits_q)
            
                # pred_q[pred_q >= 0.5] = 1
                # pred_q[pred_q < 0.5] = 0

               # print(logits_q)
               # logits_q_e=1
                pred_q=torch.ge(logits_q, 0).type(torch.cuda.FloatTensor)
               # pred_q[pred_q >= 0] = 1
               # pred_q[pred_q < 0] = 0 
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                # correct = torch.eq(pred_q, y_qry[i]).sum().item()
                # correct = sum(torch.eq(pred_q, y_qry[i]).all(axis=1))
                # correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0]+ correct
             #   print("maml-126")
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights)
                # pred_q = self.net(x_qry[i], fast_weights)
                loss_q = criterion(logits_q, y_qry[i])
                # loss_q = criterion(pred_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                #pred_q = F.sigmoid(logits_q)
                # pred_q[pred_q >= 0.5] = 1
                # pred_q[pred_q < 0.5] = 0

                pred_q=torch.ge(logits_q, 0).type(torch.cuda.FloatTensor)
               # pred_q[pred_q >= 0] = 1
               # pred_q[pred_q < 0] = 0 
                correct = torch.eq(pred_q, y_qry[i]).sum().item()       

                # correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct
          #      print("maml-146")

            for k in range(1, self.update_step):
                ##### print(k, '-th update started')

                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights)
                # pred = self.net(x_spt[i], fast_weights)
                loss = criterion(logits, y_spt[i])
                # loss = criterion(pred, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights)
                # pred_q = self.net(x_qry[i], fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = criterion(logits_q, y_qry[i])
                # loss_q = criterion(pred_q, y_qry[i])
                losses_q[k + 1] += loss_q
            #    print("maml-166")
                
                with torch.no_grad():
                    #pred_q = F.sigmoid(logits_q)
                    # pred_q[pred_q >= 0.5] = 1
                    # pred_q[pred_q < 0.5] = 0
                #    logits_q_e=1
                 #   print(len(logits_q))
                  #  print(len(logits_q[0]))
                   # print(logits_q)
                    pred_q=torch.ge(logits_q, 0).type(torch.cuda.FloatTensor)
                   # pred_q[pred_q >= 0] = 1
                   # pred_q[pred_q < 0] = 0 
                    #print(logits_q)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()       
                    # correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
               #     print("maml-177")



        # end of all tasks
        # sum over all losses on query set across all tasks
        
       # print("loss_q:",loss_q)
        
        loss_q = losses_q[-1] / task_num
    #    print("maml-188")
        # optimize theta parameters
        self.meta_optim.zero_grad()
     #   print("maml-190")
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()
     #   print("maml-193")
        # bit_size = 16
        accs = np.array(corrects) / (querysz * task_num * 16)
     #   print("accs_in_MM:", accs,corrects,querysz,task_num)
    #    print("maml-196")
        return accs



    def finetunning(self, x_spt, y_spt, x_qry, y_qry): # NOT edited yet
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        # assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        real_corrects = [0 for _ in range(self.update_step_test + 1)]

        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.BCELoss()

        # Change from Long to Float
        y_spt = y_spt.type(torch.cuda.FloatTensor)
        y_qry = y_qry.type(torch.cuda.FloatTensor)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)#net: dc-LSTM
        # pred = net(x_spt)
        loss = criterion(logits, y_spt)
        # loss = criterion(pred, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters())
            # pred_q = net(x_qry, net.parameters())
            # [setsz]
            # pred_q = F.sigmoid(logits_q)
            # pred_q[pred_q >= 0.5] = 1
            # pred_q[pred_q < 0.5] = 0
            logits_q[logits_q >= 0] = 1
            logits_q[logits_q < 0] = 0     
            correct = torch.eq(logits_q, y_qry).sum().item()   
            real_correct = torch.eq(logits_q, y_qry).all(axis=1).sum().item()
            # scalar 
            
            # correct = torch.eq(pred_q, y_qry).sum().item()

            ### This one should be correct (all 16 posibilities)
            # real_correct = torch.eq(pred_q, y_qry).all(axis=1).sum().item()
            
            
            # correct = sum(torch.eq(pred_q, y_qry).all(axis=1))
            corrects[0] = corrects[0] + correct
            real_corrects[0] = real_corrects[0] + real_correct


        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights)
            # pred_q = net(x_qry, fast_weights)
            # [setsz]
            #pred_q = F.sigmoid(logits_q)
            #pred_q = pred_q.cuda().round()
            #pred_q[pred_q >= 0.5] = 1
            #pred_q[pred_q < 0.5] = 0
            logits_q[logits_q >= 0] = 1
            logits_q[logits_q < 0] = 0     
            correct = torch.eq(logits_q, y_qry).sum().item()   
            real_correct = torch.eq(logits_q, y_qry).all(axis=1).sum().item()
            # scalar
            
            # correct = torch.eq(pred_q, y_qry).sum().item()

            # real_correct = torch.eq(pred_q, y_qry).all(axis=1).sum().item()

            ### This one should be correct (all 16 posibilities)
            ### correct = torch.eq(pred_q, y_qry).all(axis=1).sum().item()
            # correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct
            real_corrects[1] = real_corrects[1] + real_correct

# weight update
# fast_weights is the trained initial for LSTM model
        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights)
            loss = criterion(logits, y_spt)
            # pred = net(x_spt, fast_weights)
            # loss = criterion(pred, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights)
            # pred_q = net(x_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = criterion(logits_q, y_qry)
            # loss_q = criterion(pred_q, y_qry)

            with torch.no_grad():
                # pred_q = F.sigmoid(logits_q)
                # pred_q[pred_q >= 0.5] = 1
                # pred_q[pred_q < 0.5] = 0

                logits_q[logits_q >= 0] = 1
                logits_q[logits_q < 0] = 0     
                correct = torch.eq(logits_q, y_qry).sum().item()   
                real_correct = torch.eq(logits_q, y_qry).all(axis=1).sum().item()

                # correct = torch.eq(pred_q, y_qry).sum().item()
                # real_correct = torch.eq(pred_q, y_qry).all(axis=1).sum().item()
                ### This one should be correct (all 16 posibilities)
                ### correct = torch.eq(pred_q, y_qry).all(axis=1).sum().item()
                # correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
                real_corrects[k + 1] = real_corrects[k + 1] + real_correct
            
        #    path="/home/pengmiao/Mywork/PAKDD/model_torch/meta-lstm.pt"
            
        #    torch.save(net,path)


        del net

        accs = np.array(corrects) / (querysz*16)

        real_accs = np.array(real_corrects) / querysz

        ### This one should be correct (all 16 posibilities)
        ### accs = np.array(corrects) / querysz

        # Fake one
        # return accs

        return real_accs




def main():
    pass


if __name__ == '__main__':
    main()
