#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import os
import torchvision.transforms as transforms
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

mpl.use('agg')

torch.backends.cudnn.enabled = False
device = 'cpu'

# Hyper-parameters

input_size = 320 # Junk
hidden_size = 50 # Junk
num_layers = 2 # Junk

num_classes = 10
batch_size = 50


batch_Size = batch_size



# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)




def data_load(data='train'):
    # trainsize = 200
    # testsize = 40

    if data == 'test':
        samples = torchvision.datasets.MNIST(root='./mnist', train=False, download=True,
                                             transform=torchvision.transforms.Compose([transforms.ToTensor(),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,))]))
        size = 250
        a, _ = torch.utils.data.random_split(samples, [size, len(samples) - size])

    else:
        samples = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=torchvision.transforms.Compose([transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
        size = 500
        a, _ = torch.utils.data.random_split(samples, [size, len(samples) - size])

    data_loader = torch.utils.data.DataLoader(a,
                                              batch_size=batch_size,
                                              shuffle=True)
    return data_loader




# Initialise and parse inputs
parser = argparse.ArgumentParser(description='PT MCMC CNN')

parser.add_argument('-n', '--net', help='Choose rnn net, "1" for RNN, "2" for GRU, "3" for LSTM', default=4, dest="net",
                    type=int) # Junk
parser.add_argument('-s', '--samples', help='Number of samples', default=500, dest="samples", type=int)
parser.add_argument('-r', '--replicas', help='Number of chains/replicas, best to have one per availble core/cpu',
                    default=10, dest="num_chains", type=int)
parser.add_argument('-t', '--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ',
                    default=2, dest="mt_val", type=int) #Junk
parser.add_argument('-swap', '--swap', help='Swap Ratio', dest="swap_ratio", default=0.02, type=float)
parser.add_argument('-b', '--burn', help='How many samples to discard before determing posteriors', dest="burn_in",
                    default=0.25, type=float)
parser.add_argument('-pt', '--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",
                    default=0.5, type=float)
parser.add_argument('-step', '--step', help='Step size for proposals (0.02, 0.05, 0.1 etc)', dest="step_size",
                    default=0.005, type=float) # Junk
parser.add_argument('-lr', '--learning_rate', help='Learning Rate for Model', dest="learning_rate",
                    default=0.01, type=float)
args = parser.parse_args()


def f(): raise Exception("Found exit()")


class Model(nn.Module):
    # Defining input size, hidden layer size, output size and batch size respectively
    def __init__(self, topo, lrate, batch_size, rnn_net='CNN'):
        super(Model, self).__init__()
        if rnn_net == 'CNN':
            #print("CNN")
            self.conv1 = nn.Conv2d(1, 32, 5, 1)
            self.conv2 = nn.Conv2d(32, 64, 5, 1)
            self.fc1 = nn.Linear(1024, 10)
            #self.fc2 = nn.Linear(128, 10)
            self.batch_size = batch_size
            self.sigmoid = nn.Sigmoid()
            self.topo = topo
            self.los = 0
            self.softmax = nn.Softmax(dim=1)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate)


    def forward(self, x):
        x = self.conv1(x)
        # print("def")
        x = F.max_pool2d(x, 2)
        # x = F.relu(x)
        # x = nn.Dropout2d(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        # x = F.relu(x)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        # print("X Shape")
        # print(x.shape)
        x = self.fc1(x)
        # x = nn.Sigmoid(x)
        # x=F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        return x


    def evaluate_proposal(self, data, w=None):
        self.los = 0
        if w is not None:
            self.loadparameters(w)
        y_pred = torch.zeros((len(data), self.batch_size))
        prob = torch.zeros((len(data), self.batch_size, self.topo[2]))
        for i, sample in enumerate(data, 0):
            inputs, labels = sample
            a = copy.deepcopy(self.forward(inputs).detach())
            _, predicted = torch.max(a.data, 1)
            y_pred[i] = predicted
            b = copy.deepcopy(a)
            prob[i] = self.softmax(b)
            loss = self.criterion(a, labels)
            self.los += loss
        return y_pred, prob

    def langevin_gradient(self, x, w=None):
        if w is not None:
            self.loadparameters(w)
        # only one epoch
        self.los = 0
        # print(self.state_dict()['fc.weight'][0])
        for i, sample in enumerate(x, 0):
            inputs, labels = sample
            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # if (i % 50 == 0):
            # print(loss.item(), ' is loss', i)
            self.los += copy.deepcopy(loss.item())
        # print(lo,' is loss')
        return copy.deepcopy(self.state_dict())

    # returns a np arraylist of weights and biases -- layerwise
    # in order i.e. weight and bias of input and hidden then weight and bias for hidden to out
    def getparameters(self, w=None):
        l = np.array([1, 2])
        dic = {}
        if w is None:
            dic = self.state_dict()
        else:
            dic = copy.deepcopy(w)
        for name in sorted(dic.keys()):
            l = np.concatenate((l, np.array(copy.deepcopy(dic[name])).reshape(-1)), axis=None)
        l = l[2:]
        return l

    # input a dictionary of same dimensions
    def loadparameters(self, param):
        self.load_state_dict(param)

    def dictfromlist(self, param):
        dic = {}
        i = 0
        for name in sorted(self.state_dict().keys()):
            dic[name] = torch.FloatTensor(param[i:i + (self.state_dict()[name]).view(-1).shape[0]]).view(
                self.state_dict()[name].shape)
            i += (self.state_dict()[name]).view(-1).shape[0]
        # self.loadparameters(dic)
        return dic

    # input weight dictionary, mean, std dev
    def addnoiseandcopy(self, mea, std_dev):
        dic = {}
        w = self.state_dict()
        for name in (w.keys()):
            dic[name] = copy.deepcopy(w[name]) + torch.zeros(w[name].size()).normal_(mean=mea, std=std_dev)
        self.loadparameters(dic)
        return dic


class ptReplica(multiprocessing.Process):

    def __init__(self, use_langevin_gradients, learn_rate, w, minlim_param, maxlim_param, samples, traindata, testdata,
                 topology, burn_in, temperature, swap_interval, path, parameter_queue, main_process, event, batch_size,
                 rnn_net):
        # MULTIPROCESSING VARIABLES
        self.rnn = Model(topology, learn_rate, batch_size, rnn_net=rnn_net)
        multiprocessing.Process.__init__(self)
        self.processID = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event = event
        self.batch_size = batch_size
        self.temperature = temperature
        self.adapttemp = temperature
        self.swap_interval = swap_interval
        self.path = path
        self.burn_in = burn_in
        # FNN CHAIN VARIABLES (MCMC)
        self.samples = samples
        self.topology = topology
        self.traindata = traindata
        self.testdata = testdata
        self.w = w
        self.minY = np.zeros((1, 1))
        self.maxY = np.zeros((1, 1))
        self.minlim_param = minlim_param
        self.maxlim_param = maxlim_param
        self.use_langevin_gradients = use_langevin_gradients
        self.sgd_depth = 1  # always should be 1
        self.learn_rate = learn_rate
        self.l_prob = 0.5  # can be evaluated for diff problems - if data too large keep this low value since the gradients cost comp time
        # self.rnn = rnn

    def rmse(self, pred, actual):
        return self.rnn.los.item()
        # return torch.sqrt(torch.mean((pred-actual)**2))

    ''' check accuracy ashray'''
    ''' what does fxtrain_samples is used for '''
    '''    def accuracy(self,pred,actual):
            count = 0
            for i in range(pred.shape[0]):
                for j in range(self.batch_size):
                    if pred[i][j] == actual[i][j]:
                        count+=1 
            #print(count,pred.shape[0], 'is accuracy')
            return 100*(count/pred.reshape(-1).shape[0])
    '''

    def accuracy(self, data):
        # Test the model
        correct = 0
        total = 0
        for images, labels in data:
            # images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = self.rnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def likelihood_func(self, rnn, data, w=None):
        y = torch.zeros((len(data), self.batch_size))
        for i, dat in enumerate(data, 0):
            inputs, labels = dat
            y[i] = labels
        if w is not None:
            fx, prob = rnn.evaluate_proposal(data, w)
        else:
            fx, prob = rnn.evaluate_proposal(data)
        # rmse = self.rmse(fx,y)
        rmse = copy.deepcopy(self.rnn.los) / len(data)
        lhood = 0
        for i in range(len(data)):
            for j in range(self.batch_size):
                for k in range(self.topology[2]):
                    if k == y[i][j]:
                        if prob[i, j, k] == 0:
                            lhood += 0
                        else:
                            lhood += np.log(prob[i, j, k])
        return [lhood/self.adapttemp, fx, rmse]

    def prior_likelihood(self, sigma_squared, w_list):
        part1 = -1 * ((len(w_list)) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w_list)))
        log_loss = part1 - part2
        return log_loss

    def run(self):
        samples = self.samples
        batch_save = 10  # batch to append to file
        rnn = self.rnn  # Model(self.topology,self.learn_rate, rnn_net=self.rnn_net)
        # Random Initialisation of weights
        w = rnn.state_dict()
        w_size = len(rnn.getparameters(w))
        rmse_train  = np.zeros(samples)
        rmse_test = np.zeros(samples)
        acc_train = np.zeros(samples)
        acc_test = np.zeros(samples)

        weight_array = np.zeros(samples)
        weight_array1 = np.zeros(samples)
        weight_array2 = np.zeros(samples)
        sum_value_array = np.zeros(samples)

        learn_rate = self.learn_rate

        eta = 0  # Junk variable

        w_proposal = np.random.randn(w_size)
        w_proposal = rnn.dictfromlist(w_proposal)

        # Randomwalk Steps
        step_w = 0.005

        # Declare FNN

        train = self.traindata  # data_load(data='train')
        test = self.testdata  # data_load(data= 'test')

        # print(len(train),len(test),samples)
        # Evaluate Proposals
        # localtime = time.time()
        # pred_train, prob_train = rnn.evaluate_proposal(train)
        # pred_test, prob_test = rnn.evaluate_proposal(test)
        # print(time.time() - localtime, ' time for 1 train and test')
        # Check Variance of Proposal

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        # sigma_diagmat = np.zeros((w_size, w_size))  # for Equation 9 in Ref [Chandra_ICONIP2017]
        # np.fill_diagonal(sigma_diagmat, step_w)
        delta_likelihood = 0.5  # an arbitrary position
        prior_current = self.prior_likelihood(sigma_squared, rnn.getparameters(w))  # takes care of the gradients

        t1 = time.time()
        # Evaluate Likelihoods
        [likelihood, pred_train, rmsetrain] = self.likelihood_func(rnn, train)
        [_, pred_test, rmsetest] = self.likelihood_func(rnn, test)

        #print(time.time() - t1, ' time elapsed in sec for likelihood')

        # Beginning Sampling using MCMC RANDOMWALK
        y_test = torch.zeros((len(test), self.batch_size))
        for i, dat in enumerate(test, 0):
            inputs, labels = dat
            y_test[i] = copy.deepcopy(labels)
        y_train = torch.zeros((len(train), self.batch_size))
        for i, dat in enumerate(train, 0):
            inputs, labels = dat
            y_train[i] = copy.deepcopy(labels)

        trainacc = 0
        testacc = 0

        # prop_list = np.zeros((samples,w_size))
        # likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood
        # likeh_list[0,:] = [-100, -100] # to avoid prob in calc of 5th and 95th percentile later
        # surg_likeh_list = np.zeros((samples,2))
        # accept_list = np.zeros(samples)

        num_accepted = 0
        langevin_count = 0
        pt_samples = samples * 0.6  # this means that PT in canonical form with adaptive temp will work till pt  samples are reached
        init_count = 0
        #pos_w[0] = rnn.getparameters(w_proposal)
        #fxtrain_samples[i + 1,] = fxtrain_samples[i,]
        #fxtest_samples[i + 1,] = fxtest_samples[i,]
        rmse_train[0] = rmsetrain
        rmse_test[0] = rmsetest
        #acc_train[0] = self.accuracy(pred_train, y_train)
        #acc_test[0] = self.accuracy(pred_test,y_test)

        acc_train[0] = self.accuracy(train)
        acc_test[0] = self.accuracy(test)

        weight_array[0] = 0
        weight_array1[0] = 0
        weight_array2[0] = 0
        sum_value_array[0] = 0

        #acc_train[0] = 50.0
        #acc_test[0] = 50.0

        # print('i and samples')
        for i in range(samples):  # Begin sampling --------------------------------------------------------------------------

            ratio = ((samples - i) / (samples * 1.0))
            if i < pt_samples:
                self.adapttemp = self.temperature  # * ratio  #  T1=T/log(k+1);
            if i == pt_samples and init_count == 0:  # move to MCMC canonical
                self.adapttemp = 1
                [likelihood, pred_train, rmsetrain] = self.likelihood_func(rnn, train, w)
                [_, pred_test, rmsetest] = self.likelihood_func(rnn, test, w)
                init_count = 1


            lx = np.random.uniform(0, 1, 1)
            old_w = rnn.state_dict()
            if (self.use_langevin_gradients is True) and (lx < self.l_prob):
                #print("Needed")

                #print('using langevin')
                w_gd = rnn.langevin_gradient(train)  # Eq 8
                w_proposal = rnn.addnoiseandcopy(0, step_w)  # np.random.normal(w_gd, step_w, w_size) # Eq 7
                w_prop_gd = rnn.langevin_gradient(train)
                # first = np.log(multivariate_normal.pdf(w , w_prop_gd , sigma_diagmat))
                # second = np.log(multivariate_normal.pdf(w_proposal , w_gd , sigma_diagmat)) # this gives numerical instability - hence we give a simple implementation next that takes out log
                wc_delta = (rnn.getparameters(w) - rnn.getparameters(w_prop_gd))
                wp_delta = (rnn.getparameters(w_proposal) - rnn.getparameters(w_gd))
                sigma_sq = step_w
                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq
                diff_prop = first - second
                diff_prop = diff_prop / self.adapttemp
                langevin_count = langevin_count + 1
            else:
                #print("Not needed")
                diff_prop = 0
                w_proposal = rnn.addnoiseandcopy(0, step_w)  # np.random.normal(w, step_w, w_size)



            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(rnn, train)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(rnn, test)


            prior_prop = self.prior_likelihood(sigma_squared, rnn.getparameters(w_proposal))  # takes care of the gradients
            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior_current

            try:
                mh_prob = min(1, math.exp(diff_likelihood + diff_prior + diff_prop))
            except OverflowError as e:
                mh_prob = 1
            # print(likelihood_proposal,likelihood, mh_prob)

            sum_value = diff_likelihood + diff_prior + diff_prop
            u = np.log(random.uniform(0, 1))

            sum_value_array[i] = sum_value


            if u < sum_value:
                num_accepted = num_accepted + 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = copy.deepcopy(w_proposal)  # rnn.getparameters(w_proposal)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                print (i, rmsetrain, rmsetest, acc_train1, acc_test1, 'accepted')
                rmse_train[i] = rmsetrain
                rmse_test[i] = rmsetest
                acc_train[i,] = acc_train1
                acc_test[i,] = acc_test1
                # x = x + 1
            else:
                w = old_w
                rnn.loadparameters(w)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                print (i, rmsetrain, rmsetest, acc_train1, acc_test1, 'rejected')
                rmse_train[i,] = rmse_train[i-1,]
                rmse_test[i,] = rmse_test[i-1,]
                acc_train[i,] = acc_train[i-1,]
                acc_test[i,] = acc_test[i-1,]

            ll = rnn.getparameters()
            weight_array[i] = ll[0]
            weight_array1[i] = ll[100]
            weight_array2[i] = ll[50000]

            if (i + 1) % self.swap_interval == 0:
                param = np.concatenate([np.asarray([rnn.getparameters(w)]).reshape(-1), np.asarray([eta]).reshape(-1),np.asarray([likelihood]), np.asarray([self.temperature]), np.asarray([i])])
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                self.event.wait()
                result = self.parameter_queue.get()
                w = rnn.dictfromlist(result[0:w_size])
                eta = result[w_size]

            if i % 100 == 0:
                print(i, rmsetrain, rmsetest, 'Iteration Number and RMSE Train & Test')

        param = np.concatenate([np.asarray([rnn.getparameters(w)]).reshape(-1), np.asarray([eta]).reshape(-1), np.asarray([likelihood]),np.asarray([self.temperature]), np.asarray([i])])
        # print('SWAPPED PARAM',self.temperature,param)
        # self.parameter_queue.put(param)
        self.signal_main.set()
        # param = np.concatenate([s_pos_w[i-self.surrogate_interval:i,:],lhood_list[i-self.surrogate_interval:i,:]],axis=1)
        # self.surrogate_parameterqueue.put(param)


        print ((num_accepted * 100 / (samples * 1.0)), '% was Accepted')
        accept_ratio = num_accepted / (samples * 1.0) * 100

        print ((langevin_count * 100 / (samples * 1.0)), '% was Langevin')
        langevin_ratio = langevin_count / (samples * 1.0) * 100

        print('Exiting the Thread',self.temperature)

        #file_name = self.path+'/posterior/pos_w/'+'chain_'+ str(self.temperature)+ '.txt'
        #np.savetxt(file_name,pos_w )

        #file_name = self.path+'/predictions/fxtrain_samples_chain_'+ str(self.temperature)+ '.txt'
        #np.savetxt(file_name, fxtrain_samples, fmt='%1.2f')

        #file_name = self.path+'/predictions/fxtest_samples_chain_'+ str(self.temperature)+ '.txt'
        #np.savetxt(file_name, fxtest_samples, fmt='%1.2f')
        """
        pathl = self.path + '/predications' + '/chain_' + str(self.temperature)

        file_name = pathl + '/sum_value.txt'
        np.savetxt(file_name, sum_value_array, fmt='%1.2f')

        file_name = pathl + '/weight[0].txt'
        np.savetxt(file_name, weight_array, fmt='%1.2f')

        file_name = pathl + '/weight[100].txt'
        np.savetxt(file_name, weight_array1, fmt='%1.2f')

        file_name = pathl + '/weight[50000].txt'
        np.savetxt(file_name, weight_array2, fmt='%1.2f')

        file_name = pathl + '/rmse_test.txt'
        np.savetxt(file_name, rmse_test, fmt='%1.2f')

        file_name = pathl + '/rmse_train.txt'
        np.savetxt(file_name, rmse_train, fmt='%1.2f')

        file_name = pathl + '/acc_test.txt'
        np.savetxt(file_name, acc_test, fmt='%1.2f')

        file_name = pathl + '/acc_train.txt'
        np.savetxt(file_name, acc_train, fmt='%1.2f')
        """

        file_name = self.path + '/predictions/sum_value_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, sum_value_array, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[0]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[100]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array1, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[50000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array2, fmt='%1.2f')

        file_name = self.path + '/predictions/rmse_test_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, rmse_test, fmt='%1.2f')

        file_name = self.path + '/predictions/rmse_train_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, rmse_train, fmt='%1.2f')

        file_name = self.path + '/predictions/acc_test_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, acc_test, fmt='%1.2f')

        file_name = self.path + '/predictions/acc_train_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, acc_train, fmt='%1.2f')



class ParallelTempering:

    def __init__(self, use_langevin_gradients, learn_rate, topology, num_chains, maxtemp, NumSample, swap_interval,
                 path, batch_size, bi, rnn_net='RNN'):
        rnn = Model(topology, learn_rate, batch_size, rnn_net=rnn_net)
        # FNN Chain variables
        self.rnn = rnn
        self.rnn_net = rnn_net
        self.traindata = data_load(data='train')
        self.testdata = data_load(data='test')
        self.topology = topology
        self.num_param = len(rnn.getparameters(
            rnn.state_dict()))  # (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        # Parallel Tempering variables
        self.swap_interval = swap_interval
        self.path = path
        self.maxtemp = maxtemp
        self.num_swap = 0
        self.total_swap_proposals = 0
        self.num_chains = num_chains
        self.chains = []
        self.temperatures = []
        self.NumSamples = int(NumSample / self.num_chains)
        self.sub_sample_size = max(1, int(0.05 * self.NumSamples))
        # create queues for transfer of parameters between process chain
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()
        self.wait_chain = [multiprocessing.Event() for i in range(self.num_chains)]
        self.event = [multiprocessing.Event() for i in range(self.num_chains)]
        self.all_param = None
        self.geometric = True  # True (geometric)  False (Linear)
        self.minlim_param = 0.0
        self.maxlim_param = 0.0
        self.minY = np.zeros((1, 1))
        self.maxY = np.ones((1, 1))
        self.model_signature = 0.0
        self.learn_rate = learn_rate
        self.use_langevin_gradients = use_langevin_gradients
        self.batch_size = batch_size
        self.masternumsample = NumSample
        self.burni=bi

    def default_beta_ladder(self, ndim, ntemps,
                            Tmax):  # https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.

        """
        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')

        tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                        2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                        2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                        1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                        1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                        1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                        1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                        1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                        1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                        1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                        1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                        1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                        1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                        1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                        1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                        1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                        1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                        1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                        1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                        1.26579, 1.26424, 1.26271, 1.26121,
                        1.25973])

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0 * np.sqrt(np.log(4.0)) / np.sqrt(ndim)
        else:
            tstep = tstep[ndim - 1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                 'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas

    def assign_temperatures(self):
        if self.geometric == True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)
            for i in range(0, self.num_chains):
                self.temperatures.append(np.inf if betas[i] == 0 else 1.0 / betas[i])
                # print (self.temperatures[i])
        else:

            tmpr_rate = (self.maxtemp / self.num_chains)
            temp = 1
            for i in range(0, self.num_chains):
                self.temperatures.append(temp)
                temp += tmpr_rate
                # print(self.temperatures[i])

    def initialize_chains(self, burn_in):
        self.burn_in = burn_in
        self.assign_temperatures()
        self.minlim_param = np.repeat([-100], self.num_param)  # priors for nn weights
        self.maxlim_param = np.repeat([100], self.num_param)
        for i in range(0, self.num_chains):
            w = np.random.randn(self.num_param)
            w = self.rnn.dictfromlist(w)
            self.chains.append(
                ptReplica(self.use_langevin_gradients, self.learn_rate, w, self.minlim_param, self.maxlim_param,
                          self.NumSamples, self.traindata, self.testdata, self.topology, self.burn_in,
                          self.temperatures[i], self.swap_interval, self.path, self.parameter_queue[i],
                          self.wait_chain[i], self.event[i], self.batch_size, self.rnn_net))

    def surr_procedure(self, queue):
        if queue.empty() is False:
            return queue.get()
        else:
            return

    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        #        if parameter_queue_2.empty() is False and parameter_queue_1.empty() is False:
        param1 = parameter_queue_1.get()
        param2 = parameter_queue_2.get()
        w1 = param1[0:self.num_param]
        eta1 = param1[self.num_param]
        lhood1 = param1[self.num_param + 1]
        T1 = param1[self.num_param + 2]
        w2 = param2[0:self.num_param]
        eta2 = param2[self.num_param]
        lhood2 = param2[self.num_param + 1]
        T2 = param2[self.num_param + 2]
        # print('yo')
        # SWAPPING PROBABILITIES
        try:
            swap_proposal = min(1, 0.5 * np.exp(lhood2 - lhood1))
        except OverflowError:
            swap_proposal = 1
        u = np.random.uniform(0, 1)
        if u < swap_proposal:
            swapped = True
            self.total_swap_proposals += 1
            self.num_swap += 1
            param_temp = param1
            param1 = param2
            param2 = param_temp
        else:
            swapped = False
            self.total_swap_proposals += 1
        return param1, param2, swapped

    def run_chains(self):
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        # swap_proposal = np.ones(self.num_chains-1)
        # create parameter holders for paramaters that will be swapped
        # replica_param = np.zeros((self.num_chains, self.num_param))
        # lhood = np.zeros(self.num_chains)
        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.NumSamples - 1
        # number_exchange = np.zeros(self.num_chains)
        # filen = open(self.path + '/num_exchange.txt', 'a')
        # RUN MCMC CHAINS
        for l in range(0, self.num_chains):
            self.chains[l].start_chain = start
            self.chains[l].end = end
        for j in range(0, self.num_chains):
            self.wait_chain[j].clear()
            self.event[j].clear()
            self.chains[j].start()
        # SWAP PROCEDURE
        swaps_affected_main = 0
        total_swaps = 0
        for i in range(int(self.NumSamples / self.swap_interval)):
            #print(i,int(self.NumSamples/self.swap_interval), 'Counting')
            count = 0
            for index in range(self.num_chains):
                if not self.chains[index].is_alive():
                    count += 1
                    self.wait_chain[index].set()
                    #print(str(self.chains[index].temperature) + " Dead" + str(index))

            if count == self.num_chains:
                break
            #print(count,'Is the Count')
            timeout_count = 0
            for index in range(0, self.num_chains):
                # print("Waiting for chain: {}".format(index+1))
                flag = self.wait_chain[index].wait()
                if flag:
                    # print("Signal from chain: {}".format(index+1))
                    timeout_count += 1

            if timeout_count != self.num_chains:
                #print("Skipping the Swap!")
                continue
            #print("Event Occured")
            for index in range(0, self.num_chains - 1):
                #print('Starting Swap')
                swapped = False
                param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],
                                                                self.parameter_queue[index + 1])
                self.parameter_queue[index].put(param_1)
                self.parameter_queue[index + 1].put(param_2)
                if index == 0:
                    if swapped:
                        swaps_affected_main += 1
                    total_swaps += 1
            for index in range(self.num_chains):
                self.wait_chain[index].clear()
                self.event[index].set()

        print("Joining Processes")

        # JOIN THEM TO MAIN PROCESS
        for index in range(0, self.num_chains):
            print('Waiting to Join ',index,self.num_chains)
            print(self.chains[index].is_alive())
            self.chains[index].join()
            print(index, 'Chain Joined')
        self.chain_queue.join()
        #pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_vec, accept_vec, accept = self.show_results()
        rmse_train, rmse_test, acc_train, acc_test, sva, wa, wa1, wa2 = self.show_results()
        print("NUMBER OF SWAPS = ", self.num_swap)
        swap_perc = self.num_swap * 100 / self.total_swap_proposals
        #return pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_vec, swap_perc, accept_vec, accept
        return rmse_train, rmse_test, acc_train, acc_test, sva, wa, wa1, wa2

    def show_results(self):
        burnin = int(self.NumSamples * self.burn_in)
        mcmc_samples = int(self.NumSamples * 0.25)
        #likelihood_rep = np.zeros((self.num_chains, self.NumSamples - burnin,2))  # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        #accept_percent = np.zeros((self.num_chains, 1))
        #accept_list = np.zeros((self.num_chains, self.NumSamples))
        #pos_w = np.zeros((self.num_chains, self.NumSamples - burnin, self.num_param))
        #fx_train_all = np.zeros((self.num_chains, self.NumSamples - burnin, len(self.traindata)))
        rmse_train = np.zeros((self.num_chains, self.NumSamples))
        acc_train = np.zeros((self.num_chains, self.NumSamples))

        #fx_test_all = np.zeros((self.num_chains, self.NumSamples - burnin, len(self.testdata)))
        rmse_test = np.zeros((self.num_chains, self.NumSamples))
        acc_test = np.zeros((self.num_chains, self.NumSamples))
        sum_val_array=np.zeros((self.num_chains, self.NumSamples))

        weight_ar=np.zeros((self.num_chains, self.NumSamples))
        weight_ar1=np.zeros((self.num_chains, self.NumSamples))
        weight_ar2=np.zeros((self.num_chains, self.NumSamples))


        for i in range(self.num_chains):

            #file_name = self.path + '/posterior/pos_w/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            #print(self.path)
            #print(file_name)
            #dat = np.loadtxt(file_name)
            #pos_w[i, :, :] = dat[burnin:, :]

            #file_name = self.path + '/posterior/pos_likelihood/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            #dat = np.loadtxt(file_name)
            #likelihood_rep[i, :] = dat[burnin:]

            #file_name = self.path + '/posterior/accept_list/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            #dat = np.loadtxt(file_name)
            #accept_list[i, :] = dat

            file_name = self.path + '/predictions/rmse_test_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            rmse_test[i, :] = dat

            file_name = self.path + '/predictions/rmse_train_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            rmse_train[i, :] = dat

            file_name = self.path + '/predictions/acc_test_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            acc_test[i, :] = dat

            file_name = self.path + '/predictions/acc_train_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            acc_train[i, :] = dat

            file_name = self.path + '/predictions/sum_value_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            sum_val_array[i, :] = dat


            file_name = self.path + '/predictions/weight[0]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar[i, :] = dat

            file_name = self.path + '/predictions/weight[100]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar1[i, :] = dat

            file_name = self.path + '/predictions/weight[50000]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar2[i, :] = dat

        chain1_rmsetest = rmse_test[0, :]  # to get posterior of chain 0 only (PT chain with temp 1)
        chain1_rmsetrain = rmse_train[0, :]

        chain1_acctest = acc_test[0, :]
        chain1_acctrain = acc_train[0, :]

        #posterior = pos_w.transpose(2, 0, 1).reshape(self.num_param, -1)

        # fx_train = fx_train_all.transpose(2,0,1).reshape(self.traindata.shape[0],-1)  # need to comment this if need to save memory
        # fx_test = fx_test_all.transpose(2,0,1).reshape(self.testdata.shape[0],-1)

        # fx_test = fxtest_samples.reshape(self.num_chains*(self.NumSamples - burnin), self.testdata.shape[0]) # konarks version

        #likelihood_vec = likelihood_rep.transpose(2, 0, 1).reshape(2, -1)

        '''rmse_train = rmse_train[  : , 0: mcmc_samples]
        rmse_test = rmse_test[  : , 0: mcmc_samples]

        acc_train = rmse_train[  : , 0: mcmc_samples]
        acc_test = rmse_test[  : , 0: mcmc_samples] '''


        rmse_train = rmse_train.reshape((self.num_chains * self.NumSamples), 1)
        acc_train = acc_train.reshape((self.num_chains * self.NumSamples), 1)
        rmse_test = rmse_test.reshape((self.num_chains * self.NumSamples), 1)
        acc_test = acc_test.reshape((self.num_chains * self.NumSamples), 1)
        sum_val_array = sum_val_array.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar = weight_ar.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar1 = weight_ar1.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar2 = weight_ar2.reshape((self.num_chains * self.NumSamples), 1)

        x = np.linspace(0, int(self.masternumsample - self.masternumsample * self.burni), num=int(self.masternumsample - self.masternumsample * self.burni))
        x1 = np.linspace(0, self.masternumsample, num=self.masternumsample)

        path = 'mnist_torch/CNN/graphs'

        plt.plot(x1, weight_ar, label='Weight[0]')
        plt.legend(loc='upper right')
        plt.title("Weight[0] Trace")
        plt.savefig(path + '/weight[0]_samples.png')
        plt.clf()

        plt.plot(x1, weight_ar1, label='Weight[100]')
        plt.legend(loc='upper right')
        plt.title("Weight[100] Trace")
        plt.savefig(path + '/weight[100]_samples.png')
        plt.clf()

        plt.plot(x1, weight_ar2, label='Weight[50000]')
        plt.legend(loc='upper right')
        plt.title("Weight[50000] Trace")
        plt.savefig(path + '/weight[50000]_samples.png')
        plt.clf()



        plt.plot(x1, sum_val_array, label='Sum_Value')
        plt.legend(loc='upper right')
        plt.title("Sum Value Over Samples")
        plt.savefig(path + '/sum_value_samples.png')
        plt.clf()

        # plt.plot(x, acc_train, label='Train')
        # plt.legend(loc='upper right')
        # plt.title("Accuracy Train Values Over Samples")
        # plt.savefig('mnist_torch_single_chain' + '/accuracy_samples.png')
        # plt.clf()

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Accuracy Train', color=color)
        ax1.plot(x1, acc_train, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Accuracy Test', color=color)  # we already handled the x-label with ax1
        ax2.plot(x1, acc_test, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # ax3=ax1.twinx()

        # color = 'tab:green'
        # ax3.set_ylabel('Accuracy Test', color=color)  # we already handled the x-label with ax1
        # ax3.plot(x, acc_test, color=color)
        # ax3.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(path + '/superimposed_acc.png')
        plt.clf()

        fig1, ax4 = plt.subplots()

        color = 'tab:red'
        ax4.set_xlabel('Samples')
        ax4.set_ylabel('RMSE Train', color=color)
        ax4.plot(x1, rmse_train, color=color)
        ax4.tick_params(axis='y', labelcolor=color)

        ax5 = ax4.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax5.set_ylabel('RMSE Test', color=color)  # we already handled the x-label with ax1
        ax5.plot(x1, rmse_test, color=color)
        ax5.tick_params(axis='y', labelcolor=color)

        # ax6 = ax4.twinx()

        # color = 'tab:green'
        # ax6.set_ylabel('RMSE Test', color=color)  # we already handled the x-label with ax1
        # ax6.plot(x, rmse_test, color=color)
        # ax6.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(path + '/superimposed_rmse.png')
        plt.clf()


        '''rmse_train = rmse_train.reshape(self.num_chains*(mcmc_samples), 1)
        acc_train = acc_train.reshape(self.num_chains*(mcmc_samples), 1)
        rmse_test = rmse_test.reshape(self.num_chains*(mcmc_samples), 1)
        acc_test = acc_test.reshape(self.num_chains*(mcmc_samples), 1) 

        rmse_train = np.append(rmse_train, chain1_rmsetrain)
        rmse_test = np.append(rmse_test, chain1_rmsetest)  
        acc_train = np.append(acc_train, chain1_acctrain)
        acc_test = np.append(acc_test, chain1_acctest) '''

        #accept_vec = accept_list

        #accept = np.sum(accept_percent) / self.num_chains

        # np.savetxt(self.path + '/pos_param.txt', posterior.T)  # tcoment to save space

        #np.savetxt(self.path + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')

        #np.savetxt(self.path + '/accept_list.txt', accept_list, fmt='%1.2f')

        #np.savetxt(self.path + '/acceptpercent.txt', [accept], fmt='%1.2f')

        #return posterior, fx_train_all, fx_test_all, rmse_train, rmse_test, acc_train, acc_test, likelihood_vec.T, accept_vec, accept
        return rmse_train, rmse_test, acc_train, acc_test, sum_val_array, weight_ar, weight_ar1, weight_ar2

    def make_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


def main():

    topology = [input_size, hidden_size, num_classes]

    net1 = 'CNN'
    numSamples = args.samples
    batch_size = batch_Size
    num_chains = args.num_chains
    swap_ratio = args.swap_ratio
    burn_in = args.burn_in
    learning_rate = args.learning_rate
    maxtemp = 2
    use_langevin_gradients = True  # False leaves it as Random-walk proposals. Note that Langevin gradients will take a bit more time computationally
    bi = burn_in
    swap_interval = int(swap_ratio * numSamples / num_chains)  # int(swap_ratio * (NumSample/num_chains)) #how ofen you swap neighbours. note if swap is more than Num_samples, its off



    # learn_rate = 0.01  # in case langevin gradients are used. Can select other values, we found small value is ok.


    problemfolder = 'mnist_torch/' + net1  # change this to your directory for results output - produces large datasets

    name = ""
    filename = ""


    if not os.path.exists(problemfolder + name):
        os.makedirs(problemfolder + name)
    path = (problemfolder + name)


    timer = time.time()

    pt = ParallelTempering(use_langevin_gradients, learning_rate, topology, num_chains, maxtemp, numSamples,
                           swap_interval, path, batch_size, bi, rnn_net=net1)

    directories = [path + '/predictions/', path+'/graphs/']
    for d in directories:
        pt.make_directory((filename) + d)

    pt.initialize_chains(burn_in)
    #pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_rep, swap_perc, accept_vec, accept = pt.run_chains()
    rmse_train, rmse_test, acc_train, acc_test, sum_value_array, weigh_array, weigh_array1, weigh_array2 = pt.run_chains()



    timer2 = time.time()

    # list_end = accept_vec.shape[1]
    # accept_ratio = accept_vec[:,  list_end-1:list_end]/list_end
    # accept_per = np.mean(accept_ratio) * 100
    # print(accept_per, ' accept_per')

    timetotal = (timer2 - timer) /60
    print ((timetotal), 'is the minutes taken')


    """
    # #PLOTS
    acc_tr = np.mean(acc_train [:])
    acctr_std = np.std(acc_train[:])
    acctr_max = np.amax(acc_train[:])

    acc_tes = np.mean(acc_test[:])
    acctest_std = np.std(acc_test[:])
    acctes_max = np.amax(acc_test[:])

    rmse_tr = np.mean(rmse_train[:])
    rmsetr_std = np.std(rmse_train[:])
    rmsetr_max = np.amax(acc_train[:])

    rmse_tes = np.mean(rmse_test[:])
    rmsetest_std = np.std(rmse_test[:])
    rmsetes_max = np.amax(rmse_test[:])
    """

    burnin=burn_in

    acc_tr = np.mean(acc_train[int(numSamples * burnin):])
    acctr_std = np.std(acc_train[int(numSamples * burnin):])
    acctr_max = np.amax(acc_train[int(numSamples * burnin):])

    acc_tes = np.mean(acc_test[int(numSamples * burnin):])
    acctest_std = np.std(acc_test[int(numSamples * burnin):])
    acctes_max = np.amax(acc_test[int(numSamples * burnin):])

    rmse_tr = np.mean(rmse_train[int(numSamples * burnin):])
    rmsetr_std = np.std(rmse_train[int(numSamples * burnin):])
    rmsetr_max = np.amax(acc_train[int(numSamples * burnin):])

    rmse_tes = np.mean(rmse_test[int(numSamples * burnin):])
    rmsetest_std = np.std(rmse_test[int(numSamples * burnin):])
    rmsetes_max = np.amax(rmse_test[int(numSamples * burnin):])

    # outres = open(path+'/result.txt', "a+")
    # outres_db = open(path_db+'/result.txt', "a+")
    # resultingfile = open(problemfolder+'/master_result_file.txt','a+')
    # resultingfile_db = open( problemfolder_db+'/master_result_file.txt','a+')
    # xv = name+'_'+ str(run_nb)
    print("\n\n\n\n")
    print("Train Acc (Mean, Max, Std)")
    print (acc_tr, acctr_max, acctr_std)
    print("\n")
    print("Test Acc (Mean, Max, Std)")
    print(acc_tes, acctes_max, acctest_std)
    print("\n")
    print("Train RMSE (Mean, Max, Std)")
    print(rmse_tr, rmsetr_max, rmsetr_std)
    print("\n")
    print("Test RMSE (Mean, Max, Std)")
    print(rmse_tes, rmsetes_max, rmsetest_std)

if __name__ == "__main__": main()