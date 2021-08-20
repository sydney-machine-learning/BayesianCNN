import numpy as np
from numpy import array
import copy
import pandas as pd
#data = np.genfromtxt("https://raw.githubusercontent.com/sydney-machine-learning/BayesianCNN/master/Time-Series/data/ashok_mar19_mar20.csv", delimiter =',' , )[1:,1:]
train = pd.read_csv("/home/shravan/Desktop/Sunspot/deeplearning_timeseries-master/data/Sunspot/train1.csv")
test = pd.read_csv( "/home/shravan/Desktop/Sunspot/deeplearning_timeseries-master/data/Sunspot/test1.csv")
#data = np.array(data, dtype = object)
print(train.shape)
print(test.shape)
train.drop(labels=train.columns[0], axis=1, inplace=True)
test.drop(labels=train.columns[0], axis=1, inplace=True)
#np.asarray(train).reshape(585,1,15)


import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import os
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import copy
import multiprocessing
import os
import sys
import gc
import numpy as np
import random
import time
import operator
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import pickle
from sklearn import preprocessing

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        sequence = np.asarray(sequence)
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i][0:5], sequence[i][5:15]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

n_steps_in, n_steps_out = 5,10
train_X, train_Y = split_sequence(train, n_steps_in, n_steps_out)
test_X, test_Y= split_sequence(test, n_steps_in, n_steps_out)

train_X = train_X.reshape(571,1,5)[0:570]
train_Y = train_Y.reshape(571,1,10)[0:570]
test_X = test_X.reshape(371,1,5)[0:370]
test_Y = test_Y.reshape(371,1,10)[0:370]
print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)




from torch.utils.data import TensorDataset
def data_load(data='train'):
    if data == 'test':
        # transform to torch tensor
        tensor_x = torch.Tensor(train_X)
        tensor_y = torch.Tensor(train_Y)
        a = TensorDataset(tensor_x, tensor_y)

    else:
        # transform to torch tensor
        tensor_x = torch.Tensor(test_X)
        tensor_y = torch.Tensor(test_Y)
        a = TensorDataset(tensor_x, tensor_y)

    data_loader = torch.utils.data.DataLoader(a, batch_size=batch_Size, shuffle=True)
    return data_loader

from torch import nn

device = 'cpu'

# * parameter to keep track of already run samples
samples_run = 0
load = False
# Hyper-Parameters

input_size = 5  # Junk
hidden_size = 50  # Junk
num_layers = 2  # Junk
num_classes = 10
batch_size = 10
batch_Size = batch_size
step_size = 0.005#10

class Model(nn.Module):
    
    def __init__(self, topo, lrate, batch_size, cnn_net='CNN'):
        super(Model, self).__init__()
        if cnn_net == 'CNN':
            self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            self.pool = nn.MaxPool1d(2)
            self.fc1 = nn.Linear(64,10)
            self.fc2 = nn.Linear(10,10)#n_outputs) #n_outputs = 6

            self.batch_size = batch_size
            self.sigmoid = nn.Sigmoid()
            self.topo = topo
            self.los = 0
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate)
            self.softmax = nn.Softmax(dim=1)

    # Sequence of execution for the model layers

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.sigmoid(x)
        return x
    
    #Used to apply softmax and obtain loss value
    def evaluate_proposal(self, data, w=None):
        self.los = 0
        if w is not None:
            self.loadparameters(w)
        flag = False
        y_pred = torch.zeros((len(data), self.batch_size, 10))
        for i, sample in enumerate(data, 0):
            inputs, labels = sample
            predicted = copy.deepcopy(self.forward(inputs).detach())
            # _, predicted = torch.max(a.data, 1)
            if(flag):
                y_pred = torch.cat((y_pred, predicted), dim=0)
            else:
                flag = True
                y_pred = predicted
            #y_pred[i] = predicted
            # b = copy.deepcopy(a)
            # prob[i] = self.softmax(b)
            loss = self.criterion(predicted, labels.reshape(10,10))
            #print("Predicted is ", predicted, end ="$$")
            #print("Labels  : ", labels.reshape(10,10))
            #print(loss.item(), ' is loss eval', i)
            self.los += loss
        return y_pred
    
    #Applied Langevin gradient to obtain weight proposal
    def langevin_gradient(self, x, w=None):
        if w is not None:
            self.loadparameters(w)
        self.los = 0
        for i, sample in enumerate(x, 0):
            inputs, labels = sample
            outputs = self.forward(inputs)
            # _, predicted = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels.reshape(10,10))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # if (i % 50 == 0):
            #print(loss.item(), ' is loss langevin', i)
            self.los += copy.deepcopy(loss.item())
        return copy.deepcopy(self.state_dict())
    
   # Obtain a list of the model parameters (weights and biases)

    def getparameters(self, w=None):
        l = np.array([1, 2])
        dic = {}
        if w is None:
            dic = self.state_dict()
        else:
            dic = copy.deepcopy(w)
        for name in sorted(dic.keys()):
            l = np.concatenate(
                (l, np.array(copy.deepcopy(dic[name])).reshape(-1)), axis=None)
        l = l[2:]
        return l
    

    def loadparameters(self, param):
        self.load_state_dict(param)
    
    #Converting list of model parameters to Pytorch dictionary form
    def dictfromlist(self, param):
        dic = {}
        i = 0
        for name in sorted(self.state_dict().keys()):
            dic[name] = torch.FloatTensor(param[i:i + (self.state_dict()[name]).view(-1).shape[0]]).view(
                self.state_dict()[name].shape)
            i += (self.state_dict()[name]).view(-1).shape[0]
        return dic
    
    
    def addnoiseandcopy(self, mea, std_dev):
        dic = {}
        w = self.state_dict()
        for name in (w.keys()):
            dic[name] = copy.deepcopy(
                w[name]) + torch.zeros(w[name].size()).normal_(mean=mea, std=std_dev)
        self.loadparameters(dic)
        return dic
    
    
class MCMC:
    def __init__(self, samples, topology, use_langevin_gradients, lr, batch_size):
        self.samples = samples
        self.topology = topology
        self.lr = lr
        self.batch_size = batch_size
        self.use_langevin_gradients = use_langevin_gradients
        self.l_prob = 0.5
        self.cnn = Model(topology, lr, batch_size, cnn_net = 'CNN')
        self.train_data = data_load(data = 'train')
        self.test_data = data_load(data = 'test')
        self.step_size = step_size
        self.learn_rate = lr
        
    
    def rmse(self, predictions, targets):
        return self.cnn.los.item()
    
    def likelihood_func(self, cnn, data, tau_sq = 1, w=None):
        # y = torch.zeros((len(data), self.batch_size, 17))
        flag = False
        for i, dat in enumerate(data, 0):
            inputs, labels = dat
            if(flag):
                y = torch.cat((y, labels), dim=0)
            else:
                y = labels
                flag = True
        # print("loop calculated")
        if w is not None:
            fx = cnn.evaluate_proposal(data, w)
        else:
            fx = cnn.evaluate_proposal(data)
        # rmse = self.rmse(fx,y)
        # print("proposal calculated")
        rmse = copy.deepcopy(self.cnn.los) / len(data)
        #print("RMSE: ", rmse)
        loss = np.sum(-0.5*np.log(2*math.pi*tau_sq) - 0.5 *
                      np.square(y.numpy()-fx.numpy())/tau_sq)
        return [np.sum(loss) , fx, rmse] #/ self.adapttemp
    
    #Calculate prior value, change based on problem
    def prior_likelihood(self, sigma_squared, w_list):
        part1 = -1 * ((len(w_list)) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w_list)))
        log_loss = part1 - part2
        return log_loss
    
    def rmse(self,pred, actual):
        error = np.subtract(pred, actual)
        sqerror= np.sum(np.square(error))/actual.shape[0]
        return np.sqrt(sqerror)
    
    def sampler(self):
        # print("chian running")
        samples = self.samples
        cnn = self.cnn

        # Random Initialisation of weights
        w = cnn.state_dict()
        w_size = len(cnn.getparameters(w))
        step_w = self.step_size

        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)
        # acc_train = np.zeros(samples)
        # acc_test = np.zeros(samples)
        weight_array = np.zeros(samples)
        weight_array1 = np.zeros(samples)
        weight_array2 = np.zeros(samples)
        weight_array3 = np.zeros(samples)
        weight_array4 = np.zeros(samples)
        sum_value_array = np.zeros(samples)

        learn_rate = self.learn_rate
        flag = False
        for i, sample in enumerate(self.train_data, 0):
            _, label = sample
            if(flag):
                y_train = torch.cat((y_train, label), dim=0)
            else:
                flag = True
                y_train = label

        pred_train = cnn.evaluate_proposal(self.train_data)

        # flag = False
        # for i in range(len(pred)):
        #     label = pred[i]
        #     if(flag):
        #       pred_train = torch.cat((pred_train, label), dim = 0)
        #     else:
        #       flag = True
        #       pred_train = label

        step_eta = 0.2

        eta = np.log(np.var(pred_train.numpy() - y_train.numpy()))
        tau_pro = np.sum(np.exp(eta))
        # print(tau_pro)

        w_proposal = np.random.randn(w_size)
        w_proposal = cnn.dictfromlist(w_proposal)
        train = self.train_data
        test = self.test_data

        sigma_squared = 25
        prior_current = self.prior_likelihood(
            sigma_squared, cnn.getparameters(w))  # takes care of the gradients

        # Evaluate Likelihoods

        # print("calculating prob")
        [likelihood, pred_train, rmsetrain] = self.likelihood_func(
            cnn, train, tau_pro)
        # print("prior calculated")
        # print("Hi")
        [_, pred_test, rmsetest] = self.likelihood_func(cnn, test, tau_pro)

        #print("Bye")

        # Beginning sampling using MCMC

        # y_test = torch.zeros((len(test), self.batch_size))
        # for i, dat in enumerate(test, 0):
        #     inputs, labels = dat
        #     y_test[i] = copy.deepcopy(labels)
        # y_train = torch.zeros((len(train), self.batch_size))
        # for i, dat in enumerate(train, 0):
        #     inputs, labels = dat
        #     y_train[i] = copy.deepcopy(labels)

        num_accepted = 0            # TODO: save this
        langevin_count = 0  # TODO: save this

        #if(load):
         #   [langevin_count, num_accepted] = np.loadtxt(
          #      self.path+'/parameters/langevin_count_'+str(self.temperature) + '.txt')
        # TODO: remember to add number of samples from last run
        # PT in canonical form with adaptive temp will work till assigned limit
        pt_samples = (500) * 0.6
        init_count = 0

        rmse_train[0] = rmsetrain
        rmse_test[0] = rmsetest

        weight_array[0] = 0
        weight_array1[0] = 0
        weight_array2[0] = 0
        weight_array3[0] = 0
        weight_array4[0] = 0

        sum_value_array[0] = 0
        print("beginnning sampling")
        import time
        start = time.time()
        for i in range(samples):  # Begin sampling --------------------------------------------------------------------------
            # print("sampling", i)
            ratio = ((samples - i) / (samples * 1.0))   # ! why this?
            # TODO: remember to add number of samples from last run in i (i+2400<pt_samples)
            #if (i+samples_run) < pt_samples:
            #    self.adapttemp = self.temperature  # T1=T/log(k+1);
            #if i == pt_samples and init_count == 0:  # Move to canonical MCMC
                #self.adapttemp = 1
            [likelihood, pred_train, rmsetrain] = self.likelihood_func(cnn, train, tau_pro, w)
            [_, pred_test, rmsetest] = self.likelihood_func(cnn, test, tau_pro, w)
            init_count = 1

            lx = np.random.uniform(0, 1, 1)
            old_w = cnn.state_dict()

            if ((self.use_langevin_gradients is True) and (lx < self.l_prob)): #(langevin_count < self.langevin_step) or 
                #print("Length of Train ", len(train))
                w_gd = cnn.langevin_gradient(train)
                w_proposal = cnn.addnoiseandcopy(0, step_w)
                w_prop_gd = cnn.langevin_gradient(train)
                wc_delta = (cnn.getparameters(w) -
                            cnn.getparameters(w_prop_gd))
                wp_delta = (cnn.getparameters(w_proposal) -
                            cnn.getparameters(w_gd))
                sigma_sq = step_w
                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq
                diff_prop = first - second
                diff_prop = diff_prop #/ self.adapttemp
                langevin_count = langevin_count + 1
            else:
                diff_prop = 0
                w_proposal = cnn.addnoiseandcopy(0, step_w)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(
                cnn, train, tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(
                cnn, test, tau_pro)

            prior_prop = self.prior_likelihood(
                sigma_squared, cnn.getparameters(w_proposal))
            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior_current

            try:
                mh_prob = min(1, math.exp(
                    diff_likelihood + diff_prior + diff_prop))
            except OverflowError as e:
                mh_prob = 1

            sum_value = diff_likelihood + diff_prior + diff_prop
            sum_value_array[i] = sum_value
            u = (random.uniform(0, 1))
            # print(mh_prob, 'mh_prob')
            if u < mh_prob:
                num_accepted = num_accepted + 1
                likelihood = likelihood_proposal
                prior_current = prior_prop

                eta = eta_pro

                w = copy.deepcopy(w_proposal)  # cnn.getparameters(w_proposal)
                #acc_train1 = self.accuracy(train)
                #acc_test1 = self.accuracy(test)
                print(i+samples_run, rmsetrain, rmsetest, 'Accepted')
                rmse_train[i] = rmsetrain
                rmse_test[i] = rmsetest
                # acc_train[i,] = acc_train1
                # acc_test[i,] = acc_test1

            else:
                w = old_w
                cnn.loadparameters(w)
                # acc_train1 = self.accuracy(train)
                # acc_test1 = self.accuracy(test)
                print(i+samples_run, rmsetrain, rmsetest, 'Rejected')
                # implying that first proposal(i=0) will never be rejected?
                rmse_train[i, ] = rmse_train[i - 1, ]
                rmse_test[i, ] = rmse_test[i - 1, ]
                # acc_train[i,] = acc_train[i - 1,]
                # acc_test[i,] = acc_test[i - 1,]

            
            
            ll = cnn.getparameters()
            print(ll.shape)
            weight_array[i] = ll[0]
            weight_array1[i] = ll[100]
            weight_array2[i] = ll[1000]
            #weight_array3[i] = ll[4000]
            #weight_array4[i] = ll[8000]
            


        end = time.time()
        print("\n\nTotal time taken for Sampling : ", (end-start))
        print ((num_accepted * 100 / (samples * 1.0)), '% was Accepted')

        print ((langevin_count * 100 / (samples * 1.0)), '% was Langevin')
        final_preds = cnn.forward(torch.Tensor(test_X).float())
        print(cnn.forward(torch.Tensor(test_X).float()))
        print("Shape is :",final_preds.shape)
        final_preds = final_preds.detach().numpy()
        #test_Y = test_Y.reshape(370,10)
    
        for j in range(10):
                plt.figure()
                plt.plot(test_Y.reshape(370,10)[:,j], label='actual')
                plt.plot(final_preds[:,j], label='predicted')
                a = final_preds[:,j]
                b = test_Y.reshape(370,10)[:,j]
                print("RMSE for Step ",j+1,": ", self.rmse(a,b))
                plt.ylabel('RMSE')  
                plt.xlabel('Time (samples)') 
                plt.title('Actual vs Predicted')
                plt.legend()
                #plt.savefig("Results/"+name+"/"+Mname+'/pred_Step'+str(j+1)+'.png',dpi=300) 
                plt.show()
                plt.close()

        return  rmse_train, rmse_test, sum_value_array, weight_array, weight_array1, weight_array2 #acc_train, acc_test,

input_size = 320  # Junk
hidden_size = 50  # Junk
num_layers = 2  # Junk
num_classes = 5#10
batch_size = 10
batch_Size = batch_size
step_size = 0.005#10

outres = open('resultspriors.txt', 'w')

topology = [input_size, hidden_size, num_classes]

numSamples = 1000
ulg = True

learnr=0.01
burnin =0.25
mcmc = MCMC(numSamples, topology, ulg, learnr, batch_size)  # declare class
rmse_train, rmse_test, sva, wa, wa1, wa2 = mcmc.sampler() #acc_train, acc_test, 

#acc_train=acc_train[int(numSamples*burnin):]
#print(acc_train)
#acc_test=acc_test[int(numSamples*burnin):]
rmse_train=rmse_train[int(numSamples*burnin):]
rmse_test=rmse_test[int(numSamples*burnin):]
sva=sva[int(numSamples*burnin):]
#print(lpa)

print("\n\n\n\n\n\n\n\n")
print("Mean of RMSE Train")
print(np.mean(rmse_train))
print("\n")
# print("Mean of Accuracy Train")
#print(np.mean(acc_train))
#print("\n")
print("Mean of RMSE Test")
print(np.mean(rmse_test))
print("\n")
#print("Mean of Accuracy Test")
#print(np.mean(acc_test))
print ('sucessfully sampled')
problemfolder = 'timeseries_torch_single_chain'
os.makedirs(problemfolder)


x = np.linspace(0, int(numSamples-numSamples*burnin), num=int(numSamples-numSamples*burnin))
x1 = np.linspace(0, numSamples, num=numSamples)

plt.plot(x1, wa, label='Weight[0]')
plt.legend(loc='upper right')
plt.title("Weight[0] Trace")
plt.savefig('timeseries_torch_single_chain' + '/weight[0]_samples.png')
plt.clf()

plt.plot(x1, wa1, label='Weight[100]')
plt.legend(loc='upper right')
plt.title("Weight[100] Trace")
plt.savefig('timeseries_torch_single_chain' + '/weight[100]_samples.png')
plt.clf()

plt.plot(x1,wa2, label='Weight[1000]')
plt.legend(loc='upper right')
plt.title("Weight[50000] Trace")
plt.savefig('timeseries_torch_single_chain' + '/weight[1000]_samples.png')
plt.clf()
plt.plot(x, sva, label='Sum_Value')
plt.legend(loc='upper right')
plt.title("Sum Value Over Samples")
plt.savefig('timeseries_torch_single_chain'+'/sum_value_samples.png')
plt.clf()


#plt.plot(x, acc_train, label='Train')
#plt.legend(loc='upper right')
#plt.title("Accuracy Train Values Over Samples")
#plt.savefig('mnist_torch_single_chain' + '/accuracy_samples.png')
#plt.clf()
fig, ax1 = plt.subplots()

    #color = 'tab:red'
    #ax1.set_xlabel('Samples')
    #ax1.set_ylabel('Accuracy Train', color=color)
    #ax1.plot(x, acc_train, color=color)
    #ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    #color = 'tab:blue'
    #ax2.set_ylabel('Accuracy Test', color=color)  # we already handled the x-label with ax1
    #ax2.plot(x, acc_test, color=color)
    #ax2.tick_params(axis='y', labelcolor=color)

    #ax3=ax1.twinx()

    #color = 'tab:green'
    #ax3.set_ylabel('Accuracy Test', color=color)  # we already handled the x-label with ax1
    #ax3.plot(x, acc_test, color=color)
    #ax3.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('timeseries_torch_single_chain' + '/superimposed_acc.png')
plt.clf()
fig1, ax4 = plt.subplots()
color = 'tab:red'
ax4.set_xlabel('Samples')
ax4.set_ylabel('RMSE Train', color=color)
ax4.plot(x, rmse_train, color=color)
ax4.tick_params(axis='y', labelcolor=color)

ax5 = ax4.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax5.set_ylabel('RMSE Test', color=color)  # we already handled the x-label with ax1
ax5.plot(x, rmse_test, color=color)
ax5.tick_params(axis='y', labelcolor=color)

    #ax6 = ax4.twinx()

    #color = 'tab:green'
    #ax6.set_ylabel('RMSE Test', color=color)  # we already handled the x-label with ax1
    #ax6.plot(x, rmse_test, color=color)
    #ax6.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('timeseries_torch_single_chain' + '/superimposed_rmse.png')
plt.clf()
