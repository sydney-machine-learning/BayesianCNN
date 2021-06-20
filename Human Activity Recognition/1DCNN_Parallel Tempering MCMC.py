import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.utils import to_categorical

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

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'UCI HAR Dataset/UCI HAR Dataset/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'UCI HAR Dataset/UCI HAR Dataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    ty = trainy
    tty=testy
    # one hot encode y
    #trainy = to_categorical(trainy)
    #testy = to_categorical(testy)
    trainy.flatten()
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy
  trainX, trainy, testX, testy = load_dataset()
  
  #FIX THIS FUNCTION

device = 'cpu'

# * parameter to keep track of already run samples
samples_run = 0
load = False
# Hyper-Parameters

input_size = 320  # Junk
hidden_size = 50  # Junk
num_layers = 2  # Junk
num_classes = 6
batch_size = 16
batch_Size = batch_size
step_size = 10
verbose, epochs, batch_size = 0, 10, 16#32

from torch.utils.data import TensorDataset
def data_load(data='train'):
    if data == 'test':
        #transform to torch tensor
        tensor_x = torch.Tensor(testX)                     #(np.expand_dims(testX[:, :, :], axis=1))
        tensor_y = torch.Tensor(testy)
        a = TensorDataset(tensor_x,tensor_y)       #TensorDataset(tensor_x, tensor_y)

    else:
        # transform to torch tensor
        tensor_x = torch.Tensor(trainX)
        tensor_y = torch.Tensor(trainy)
        a =  TensorDataset(tensor_x, tensor_y)          #TensorDataset(tensor_x, tensor_y)

    data_loader = torch.utils.data.DataLoader(
        a, batch_size=batch_Size, shuffle=True)
    return data_loader
  
  n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

mid_feature_size = 64
mid_mlp = 100
time_kernel_size = 3
dropout = 0.5
pool_size = 2
out_maxpool = 62

class Model(nn.Module):
    def __init__(self, topo, lrate, batch_size, cnn_net='CNN'):
        super(Model, self).__init__()
        if cnn_net == 'CNN':
            self.conv1 = nn.Conv1d(in_channels = 128, out_channels = 64, kernel_size = 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            self.conv2 = nn.Conv1d(64, 64,3)
            self.dout = nn.Dropout(0.5)
            self.pool = nn.MaxPool1d(2)
            self.fc1 = nn.Linear(128,100)
            self.fc2 = nn.Linear(100, 80)
            self.fc3 = nn.Linear(80,60)
            self.fc4 = nn.Linear(60,24)
            self.fc5 = nn.Linear(24,6)#n_outputs) #n_outputs = 6
            
            #self.conv1 = nn.Conv1d(in_channels = 128, out_channels = 64, kernel_size = 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            #self.conv2 = nn.Conv1d(64, 64,3)
            #self.dout = nn.Dropout(0.5)
            #self.pool = nn.MaxPool1d(2)
            #self.fc1 = nn.Linear(out_maxpool * mid_feature_size,100)
            #self.fc2 = nn.Linear(100,n_outputs) #n_outputs = 6
            
        
            self.batch_size = batch_size
            self.sigmoid = nn.Sigmoid()
            self.topo = topo
            self.los = 0
            # self.softmax = nn.Softmax(dim=1)
            self.criterion = torch.torch.nn.CrossEntropyLoss()#nn.NLLLoss()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate)
            self.softmax = nn.Softmax(dim=1)
    
   # def sigmoid(self, z):
    #    return 1 / (1 + torch.exp(-z))
    
    def forward(self, x):

        x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.dout(x)
        x = self.pool(x)
        #x = x.reshape(x.shape[0], -1)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)

        #x = F.log_softmax(x, dim=1)
        return x
    
    def evaluate_proposal(self, data, w=None):
        self.los = 0
        if w is not None:
            self.loadparameters(w)
        y_pred = torch.zeros((len(data), self.batch_size))
        prob = torch.zeros((len(data), self.batch_size, self.topo[2]))
        for i, sample in enumerate(data, 0):
            inputs, labels = sample
            labels = labels.view(labels.size(0)).type(torch.LongTensor)
            a = copy.deepcopy(self.forward(inputs).detach())
            _, predicted = torch.max(a.data, 1)
            # y_pred[i] = torch.argmax(copy.deepcopy(a),dim=1)
            if(labels.size(0)==16):
                y_pred[i] = predicted
            # print(a)
            # print(a.shape)
            # f()
                b = copy.deepcopy(a)
                prob[i] = self.softmax(b)
            # prob[i] = self.softmax(a)
            # print(predicted.shape)
            # print(labels.shape)
            
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
            if(labels.size(0)==16):
                loss = self.criterion(outputs, labels.view(labels.size(0)).type(torch.LongTensor))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # if (i % 50 == 0):
            # print(loss.item(), ' is loss', i)
            self.los += copy.deepcopy(loss.item())
        # print(lo,' is loss')
        return copy.deepcopy(self.state_dict())

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

    def dictfromlist(self, param):
        dic = {}
        i = 0
        for name in sorted(self.state_dict().keys()):
            dic[name] = torch.FloatTensor(param[i:i + (self.state_dict()[name]).view(-1).shape[0]]).view(
                self.state_dict()[name].shape)
            i += (self.state_dict()[name]).view(-1).shape[0]
        # self.loadparameters(dic)
        return dic

    def loadparameters(self, param):
        self.load_state_dict(param)

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
                 rnn_net, step_size):
        self.rnn = Model(topology, learn_rate, batch_size)#, rnn_net=rnn_net)
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
        self.sgd_depth = 1  # Keep as 1
        self.learn_rate = learn_rate
        self.l_prob = 0.7  # Ratio of langevin based proposals, higher value leads to more computation time, evaluate for different problems
        self.step_size = step_size
        # ----------------

    def rmse(self, predictions, targets):
        return self.rnn.los.item()

    def likelihood_func(self, rnn, data, w=None):
        y = torch.zeros((len(data), 16))#self.batch_size))
        for i, dat in enumerate(data, 0):
            inputs, labels = dat
            labels = labels.view(labels.size(0)).type(torch.LongTensor)
            if(labels.size(0)==16):
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
                        if prob[i,j,k] == 0:
                            lhood+=0
                        else:
                            lhood += np.log(prob[i, j, k])
        return [lhood, fx, rmse]

    def prior_likelihood(self, sigma_squared, w_list):
        part1 = -1 * ((len(w_list)) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w_list)))
        log_loss = part1 - part2
        return log_loss


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
    
    def run(self):
        samples = self.samples
        rnn = self.rnn

        # Random Initialisation of weights
        w = rnn.state_dict()
        w_size = len(rnn.getparameters(w))
        step_w = self.step_size

        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)
        acc_train = np.zeros(samples)
        acc_test = np.zeros(samples)
        weight_array = np.zeros(samples)
        weight_array1 = np.zeros(samples)
        weight_array2 = np.zeros(samples)
        #weight_array3 = np.zeros(samples)
        #weight_array4 = np.zeros(samples)
        sum_value_array = np.zeros(samples)

        learn_rate = self.learn_rate

        eta = 0  # Junk variable



        w_proposal = np.random.randn(w_size)
        w_proposal = rnn.dictfromlist(w_proposal)
        train = self.traindata
        test = self.testdata

        sigma_squared = 25
        prior_current = self.prior_likelihood(sigma_squared, rnn.getparameters(w))  # takes care of the gradients

        # Evaluate Likelihoods



        [likelihood, pred_train, rmsetrain] = self.likelihood_func(rnn, train)

        #print("Hi")
        [_, pred_test, rmsetest] = self.likelihood_func(rnn, test)

        #print("Bye")

        # Beginning sampling using MCMC

        y_test = torch.zeros((len(test), self.batch_size))
        for i, dat in enumerate(test, 0):
            inputs, labels = dat
            if(labels.size(0)==16):
                y_test[i] = copy.deepcopy(labels.view(labels.size(0)).type(torch.LongTensor))
        y_train = torch.zeros((len(train), self.batch_size))
        for i, dat in enumerate(train, 0):
            inputs, labels = dat
            if(labels.size(0)==16):
                y_train[i] = copy.deepcopy(labels.view(labels.size(0)).type(torch.LongTensor))

        num_accepted = 0
        langevin_count = 0
        pt_samples = samples * 0.6  # PT in canonical form with adaptive temp will work till assigned limit
        init_count = 0

        rmse_train[0] = rmsetrain
        rmse_test[0] = rmsetest
        acc_train[0] = self.accuracy(train)
        acc_test[0] = self.accuracy(test)

        weight_array[0] = 0
        weight_array1[0] = 0
        weight_array2[0] = 0
        #weight_array3[0] = 0
        #weight_array4[0] = 0

        sum_value_array[0] = 0

        for i in range(
                samples):  # Begin sampling --------------------------------------------------------------------------

            ratio = ((samples - i) / (samples * 1.0))
            if i < pt_samples:
                self.adapttemp = self.temperature  # T1=T/log(k+1);
            if i == pt_samples and init_count == 0:  # Move to canonical MCMC
                self.adapttemp = 1
                [likelihood, pred_train, rmsetrain] = self.likelihood_func(rnn, train, w)
                [_, pred_test, rmsetest] = self.likelihood_func(rnn, test, w)
                init_count = 1

            lx = np.random.uniform(0, 1, 1)
            old_w = rnn.state_dict()



            if (self.use_langevin_gradients is True) and (lx < self.l_prob):
                w_gd = rnn.langevin_gradient(train)
                w_proposal = rnn.addnoiseandcopy(0, step_w)
                w_prop_gd = rnn.langevin_gradient(train)
                wc_delta = (rnn.getparameters(w) - rnn.getparameters(w_prop_gd))
                wp_delta = (rnn.getparameters(w_proposal) - rnn.getparameters(w_gd))
                sigma_sq = step_w
                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq
                diff_prop = first - second
                diff_prop = diff_prop / self.adapttemp
                langevin_count = langevin_count + 1
            else:
                diff_prop = 0
                w_proposal = rnn.addnoiseandcopy(0, step_w)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(rnn, train)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(rnn, test)

            prior_prop = self.prior_likelihood(sigma_squared, rnn.getparameters(w_proposal))
            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior_current

            """
            try:
                mh_prob = min(1, math.exp(diff_likelihood + diff_prior + diff_prop))
            except OverflowError as e:
                mh_prob = 1
            """


            sum_value = diff_likelihood + diff_prior + diff_prop
            sum_value_array[i] = sum_value
            u = np.log(random.uniform(0, 1))

            if u < sum_value:
                num_accepted = num_accepted + 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = copy.deepcopy(w_proposal)  # rnn.getparameters(w_proposal)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                print(i, rmsetrain, rmsetest, acc_train1, acc_test1, 'Accepted')
                rmse_train[i] = rmsetrain
                rmse_test[i] = rmsetest
                acc_train[i,] = acc_train1
                acc_test[i,] = acc_test1

            else:
                w = old_w
                rnn.loadparameters(w)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                print(i, rmsetrain, rmsetest, acc_train1, acc_test1, 'Rejected')
                rmse_train[i,] = rmse_train[i - 1,]
                rmse_test[i,] = rmse_test[i - 1,]
                acc_train[i,] = acc_train[i - 1,]
                acc_test[i,] = acc_test[i - 1,]

            ll = rnn.getparameters()
            weight_array[i] = ll[0]#[0]
            weight_array1[i] = ll[100]#[100]
            weight_array2[i] = ll[10000]#[50000]
            #weight_array3[i] = ll#[40000]
            #weight_array4[i] = ll#[60000]

            if (i + 1) % self.swap_interval == 0:
                param = np.concatenate([np.asarray([rnn.getparameters(w)]).reshape(-1), np.asarray([eta]).reshape(-1),
                                        np.asarray([likelihood]), np.asarray([self.temperature]), np.asarray([i])])
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                self.event.wait()
                result = self.parameter_queue.get()
                w = rnn.dictfromlist(result[0:w_size])
                eta = result[w_size]

            if i % 100 == 0:
                print(i, rmsetrain, rmsetest, 'Iteration Number and RMSE Train & Test')

        """
        big_data=data_load1()
        final_test_acc=self.accuracy(big_data)
        print(final_test_acc)
        """

        param = np.concatenate(
            [np.asarray([rnn.getparameters(w)]).reshape(-1), np.asarray([eta]).reshape(-1), np.asarray([likelihood]),
             np.asarray([self.temperature]), np.asarray([i])])
        # print('SWAPPED PARAM',self.temperature,param)
        # self.parameter_queue.put(param)
        self.signal_main.set()
        # param = np.concatenate([s_pos_w[i-self.surrogate_interval:i,:],lhood_list[i-self.surrogate_interval:i,:]],axis=1)
        # self.surrogate_parameterqueue.put(param)

        print((num_accepted * 100 / (samples * 1.0)), '% was Accepted')
        accept_ratio = num_accepted / (samples * 1.0) * 100

        print((langevin_count * 100 / (samples * 1.0)), '% was Langevin')
        langevin_ratio = langevin_count / (samples * 1.0) * 100

        print('Exiting the Thread', self.temperature)

        file_name = self.path + '/predictions/sum_value_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, sum_value_array, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[0]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[100]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array1, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[50000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array2, fmt='%1.2f')

        #file_name = self.path + '/predictions/weight[40000]_' + str(self.temperature) + '.txt'
        #np.savetxt(file_name, weight_array3, fmt='%1.2f')

        #file_name = self.path + '/predictions/weight[60000]_' + str(self.temperature) + '.txt'
        #np.savetxt(file_name, weight_array4, fmt='%1.2f')

        file_name = self.path + '/predictions/rmse_test_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, rmse_test, fmt='%1.2f')

        file_name = self.path + '/predictions/rmse_train_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, rmse_train, fmt='%1.2f')

        file_name = self.path + '/predictions/acc_test_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, acc_test, fmt='%1.2f')

        file_name = self.path + '/predictions/acc_train_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, acc_train, fmt='%1.2f')

        file_name = self.path + '/predictions/accept_percentage' + str(self.temperature) + '.txt'
        with open(file_name, 'w') as f:
            f.write('%d' % accept_ratio)

# Manages the parallel tempering, initialises and executes the parallel chains

class ParallelTempering:
    def __init__(self, use_langevin_gradients, learn_rate, topology, num_chains, maxtemp, NumSample, swap_interval,
                 path, batch_size, bi, rnn_net, step_size):
        rnn = Model(topology, learn_rate, batch_size) #,rnn_net=rnn_net)
        self.rnn = rnn
        self.rnn_net = 'CNN'#rnn_net
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
        self.burni = bi
        self.step_size = step_size

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
                          self.wait_chain[i], self.event[i], self.batch_size, self.rnn_net, self.step_size))

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
            # print(i,int(self.NumSamples/self.swap_interval), 'Counting')
            count = 0
            for index in range(self.num_chains):
                if not self.chains[index].is_alive():
                    count += 1
                    self.wait_chain[index].set()
                    # print(str(self.chains[index].temperature) + " Dead" + str(index))

            if count == self.num_chains:
                break
            # print(count,'Is the Count')
            timeout_count = 0
            for index in range(0, self.num_chains):
                # print("Waiting for chain: {}".format(index+1))
                flag = self.wait_chain[index].wait()
                if flag:
                    # print("Signal from chain: {}".format(index+1))
                    timeout_count += 1

            if timeout_count != self.num_chains:
                # print("Skipping the Swap!")
                continue
            # print("Event Occured")
            for index in range(0, self.num_chains - 1):
                # print('Starting Swap')
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
            print('Waiting to Join ', index, self.num_chains)
            print(self.chains[index].is_alive())
            self.chains[index].join()
            print(index, 'Chain Joined')
        self.chain_queue.join()
        # pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_vec, accept_vec, accept = self.show_results()
        rmse_train, rmse_test, acc_train, acc_test, apal = self.show_results()
        print("NUMBER OF SWAPS = ", self.num_swap)
        swap_perc = self.num_swap * 100 / self.total_swap_proposals
        # return pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_vec, swap_perc, accept_vec, accept
        return rmse_train, rmse_test, acc_train, acc_test, apal, swap_perc

    def show_results(self):
        burnin = int(self.NumSamples * self.burn_in)
        mcmc_samples = int(self.NumSamples * 0.25)
        # likelihood_rep = np.zeros((self.num_chains, self.NumSamples - burnin,2))  # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        # accept_percent = np.zeros((self.num_chains, 1))
        # accept_list = np.zeros((self.num_chains, self.NumSamples))
        # pos_w = np.zeros((self.num_chains, self.NumSamples - burnin, self.num_param))
        # fx_train_all = np.zeros((self.num_chains, self.NumSamples - burnin, len(self.traindata)))
        rmse_train = np.zeros((self.num_chains, self.NumSamples))
        acc_train = np.zeros((self.num_chains, self.NumSamples))

        # fx_test_all = np.zeros((self.num_chains, self.NumSamples - burnin, len(self.testdata)))
        rmse_test = np.zeros((self.num_chains, self.NumSamples))
        acc_test = np.zeros((self.num_chains, self.NumSamples))
        sum_val_array = np.zeros((self.num_chains, self.NumSamples))

        weight_ar = np.zeros((self.num_chains, self.NumSamples))
        weight_ar1 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar2 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar3 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar4 = np.zeros((self.num_chains, self.NumSamples))

        accept_percentage_all_chains = np.zeros(self.num_chains)

        for i in range(self.num_chains):
            # file_name = self.path + '/posterior/pos_w/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            # print(self.path)
            # print(file_name)
            # dat = np.loadtxt(file_name)
            # pos_w[i, :, :] = dat[burnin:, :]

            # file_name = self.path + '/posterior/pos_likelihood/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            # dat = np.loadtxt(file_name)
            # likelihood_rep[i, :] = dat[burnin:]

            # file_name = self.path + '/posterior/accept_list/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            # dat = np.loadtxt(file_name)
            # accept_list[i, :] = dat

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

            file_name = self.path + '/predictions/weight[40000]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar3[i, :] = dat

            file_name = self.path + '/predictions/weight[60000]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar4[i, :] = dat

            file_name = self.path + '/predictions/accept_percentage' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            accept_percentage_all_chains[i] = dat

        rmse_train_single_chain_plot = rmse_train[0, :]
        rmse_test_single_chain_plot = rmse_test[0, :]
        acc_train_single_chain_plot = acc_train[0, :]
        acc_test_single_chain_plot = acc_test[0, :]
        sum_val_array_single_chain_plot = sum_val_array[0]

        path = 'mnist_torch/CNN/graphs'

        x2 = np.linspace(0, self.NumSamples, num=self.NumSamples)

        plt.plot(x2, sum_val_array_single_chain_plot, label='Sum Value')
        plt.legend(loc='upper right')
        plt.title("Sum Value Single Chain")
        plt.savefig(path + '/sum_value_single_chain.png')
        plt.clf()

        color = 'tab:red'
        plt.plot(x2, acc_train_single_chain_plot, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x2, acc_test_single_chain_plot, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(path + '/superimposed_acc_single_chain.png')
        plt.clf()

        color = 'tab:red'
        plt.plot(x2, rmse_train_single_chain_plot, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x2, rmse_test_single_chain_plot, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(path + '/superimposed_rmse_single_chain.png')
        plt.clf()

        """
        fig2, ax7 = plt.subplots()
        color = 'tab:red'
        ax7.set_xlabel('Samples')
        ax7.set_ylabel('Accuracy Train Single Chain', color=color)
        ax7.plot(x2, acc_train_single_chain_plot, color=color)
        ax7.tick_params(axis='y', labelcolor=color)
        ax8 = ax7.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax8.set_ylabel('Accuracy Test Single Chain', color=color)  # we already handled the x-label with ax1
        ax8.plot(x2, acc_test_single_chain_plot, color=color)
        ax8.tick_params(axis='y', labelcolor=color)
        fig2.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(path + '/superimposed_acc_single_chain.png')
        plt.clf()
        fig3, ax9 = plt.subplots()
        color = 'tab:red'
        ax9.set_xlabel('Samples')
        ax9.set_ylabel('RMSE Train Single Chain', color=color)
        ax9.plot(x2, rmse_train_single_chain_plot, color=color)
        ax9.tick_params(axis='y', labelcolor=color)
        ax10 = ax9.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax10.set_ylabel('RMSE Test Single Chain', color=color)  # we already handled the x-label with ax1
        ax10.plot(x2, rmse_test_single_chain_plot, color=color)
        ax10.tick_params(axis='y', labelcolor=color)
        fig3.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(path + '/superimposed_rmse_single_chain.png')
        plt.clf()
        """

        rmse_train = rmse_train.reshape((self.num_chains * self.NumSamples), 1)
        acc_train = acc_train.reshape((self.num_chains * self.NumSamples), 1)
        rmse_test = rmse_test.reshape((self.num_chains * self.NumSamples), 1)
        acc_test = acc_test.reshape((self.num_chains * self.NumSamples), 1)
        sum_val_array = sum_val_array.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar = weight_ar.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar1 = weight_ar1.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar2 = weight_ar2.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar3 = weight_ar3.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar4 = weight_ar4.reshape((self.num_chains * self.NumSamples), 1)

        x = np.linspace(0, int(self.masternumsample - self.masternumsample * self.burni),
                        num=int(self.masternumsample - self.masternumsample * self.burni))
        x1 = np.linspace(0, self.masternumsample, num=self.masternumsample)

        plt.plot(x1, weight_ar, label='Weight[0]')
        plt.legend(loc='upper right')
        plt.title("Weight[0] Trace")
        plt.savefig(path + '/weight[0]_samples.png')
        plt.clf()

        plt.hist(weight_ar, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(path + '/weight[0]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar1, label='Weight[100]')
        plt.legend(loc='upper right')
        plt.title("Weight[100] Trace")
        plt.savefig(path + '/weight[100]_samples.png')
        plt.clf()

        plt.hist(weight_ar1, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(path + '/weight[100]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar2, label='Weight[10000]')
        plt.legend(loc='upper right')
        plt.title("Weight[10000] Trace")
        plt.savefig(path + '/weight[10000]_samples.png')
        plt.clf()

        plt.hist(weight_ar2, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(path + '/weight[10000]_hist.png')
        plt.clf()

        #plt.plot(x1, weight_ar3, label='Weight[40000]')
        #plt.legend(loc='upper right')
        #plt.title("Weight[40000] Trace")
        #plt.savefig(path + '/weight[40000]_samples.png')
        #plt.clf()

        #plt.hist(weight_ar3, bins=20, color="blue", alpha=0.7)
        #plt.ylabel('Frequency')
        #plt.xlabel('Parameter Values')
        #plt.savefig(path + '/weight[40000]_hist.png')
        #plt.clf()

        #plt.plot(x1, weight_ar4, label='Weight[60000]')
        #plt.legend(loc='upper right')
        #plt.title("Weight[60000] Trace")
        #plt.savefig(path + '/weight[60000]_samples.png')
        #plt.clf()

        #plt.hist(weight_ar4, bins=20, color="blue", alpha=0.7)
        #plt.ylabel('Frequency')
        #plt.xlabel('Parameter Values')
        #plt.savefig(path + '/weight[60000]_hist.png')
        #plt.clf()

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

        color = 'tab:red'
        plt.plot(x1, acc_train, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x1, acc_test, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(path + '/superimposed_acc.png')
        plt.clf()

        color = 'tab:red'
        plt.plot(x1, rmse_train, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x1, rmse_test, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(path + '/superimposed_rmse.png')
        plt.clf()

        """
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
        """

        '''rmse_train = rmse_train.reshape(self.num_chains*(mcmc_samples), 1)
        acc_train = acc_train.reshape(self.num_chains*(mcmc_samples), 1)
        rmse_test = rmse_test.reshape(self.num_chains*(mcmc_samples), 1)
        acc_test = acc_test.reshape(self.num_chains*(mcmc_samples), 1) 
        rmse_train = np.append(rmse_train, chain1_rmsetrain)
        rmse_test = np.append(rmse_test, chain1_rmsetest)  
        acc_train = np.append(acc_train, chain1_acctrain)
        acc_test = np.append(acc_test, chain1_acctest) '''

        # accept_vec = accept_list

        # accept = np.sum(accept_percent) / self.num_chains

        # np.savetxt(self.path + '/pos_param.txt', posterior.T)  # tcoment to save space

        # np.savetxt(self.path + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')

        # np.savetxt(self.path + '/accept_list.txt', accept_list, fmt='%1.2f')

        # np.savetxt(self.path + '/acceptpercent.txt', [accept], fmt='%1.2f')

        # return posterior, fx_train_all, fx_test_all, rmse_train, rmse_test, acc_train, acc_test, likelihood_vec.T, accept_vec, accept
        return rmse_train, rmse_test, acc_train, acc_test, accept_percentage_all_chains

    def make_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

def main():
    topology = [input_size, hidden_size, num_classes]

    net1 = 'CNN'
    numSamples = 4000#args.samples
    batch_size = 16#batch_Size
    num_chains = 8#args.num_chains
    swap_ratio = 0.005#args.swap_ratio
    burn_in = 0.5#args.burn_in
    learning_rate = 0.01#args.learning_rate
    step_size = 0.005#args.step_size
    maxtemp = 2
    use_langevin_gradients = True  # False leaves it as Random-walk proposals. Note that Langevin gradients will take a bit more time computationally
    bi = burn_in
    swap_interval = int(
        swap_ratio * numSamples / num_chains)  # int(swap_ratio * (NumSample/num_chains)) #how ofen you swap neighbours. note if swap is more than Num_samples, its off

    # learn_rate = 0.01  # in case langevin gradients are used. Can select other values, we found small value is ok.

    problemfolder = 'mnist_torch/' + net1  # change this to your directory for results output - produces large datasets

    name = ""
    filename = ""

    if not os.path.exists(problemfolder + name):
        os.makedirs(problemfolder + name)
    path = (problemfolder + name)

    timer = time.time()

    pt = ParallelTempering(use_langevin_gradients, learning_rate, topology, num_chains, maxtemp, numSamples,
                           swap_interval, path, batch_size, bi, net1, step_size)

    directories = [path + '/predictions/', path + '/graphs/']
    for d in directories:
        pt.make_directory((filename) + d)

    pt.initialize_chains(burn_in)
    # pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_rep, swap_perc, accept_vec, accept = pt.run_chains()
    rmse_train, rmse_test, acc_train, acc_test, accept_percent_all, sp = pt.run_chains()

    timer2 = time.time()

    # list_end = accept_vec.shape[1]
    # accept_ratio = accept_vec[:,  list_end-1:list_end]/list_end
    # accept_per = np.mean(accept_ratio) * 100
    # print(accept_per, ' accept_per')

    timetotal = (timer2 - timer) / 60

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

    burnin = burn_in

    acc_tr = np.mean(acc_train[int(numSamples * burnin):])
    acctr_std = np.std(acc_train[int(numSamples * burnin):])
    acctr_max = np.amax(acc_train[int(numSamples * burnin):])

    acc_tes = np.mean(acc_test[int(numSamples * burnin):])
    acctest_std = np.std(acc_test[int(numSamples * burnin):])
    acctes_max = np.amax(acc_test[int(numSamples * burnin):])

    rmse_tr = np.mean(rmse_train[int(numSamples * burnin):])
    rmsetr_std = np.std(rmse_train[int(numSamples * burnin):])
    rmsetr_max = np.amax(rmse_train[int(numSamples * burnin):])

    rmse_tes = np.mean(rmse_test[int(numSamples * burnin):])
    rmsetest_std = np.std(rmse_test[int(numSamples * burnin):])
    rmsetes_max = np.amax(rmse_test[int(numSamples * burnin):])

    accept_percent_mean = np.mean(accept_percent_all)

    # outres = open(path+'/result.txt', "a+")
    # outres_db = open(path_db+'/result.txt', "a+")
    # resultingfile = open(problemfolder+'/master_result_file.txt','a+')
    # resultingfile_db = open( problemfolder_db+'/master_result_file.txt','a+')
    # xv = name+'_'+ str(run_nb)
    print("\n\n\n\n")
    print("Train Acc (Mean, Max, Std)")
    print(acc_tr, acctr_max, acctr_std)
    print("\n")
    print("Test Acc (Mean, Max, Std)")
    print(acc_tes, acctes_max, acctest_std)
    print("\n")
    print("Train RMSE (Mean, Max, Std)")
    print(rmse_tr, rmsetr_max, rmsetr_std)
    print("\n")
    print("Test RMSE (Mean, Max, Std)")
    print(rmse_tes, rmsetes_max, rmsetest_std)
    print("\n")
    print("Acceptance Percentage Mean")
    print(accept_percent_mean)
    print("\n")
    print("Swap Percentage")
    print(sp)
    print("\n")
    print("Time (Minutes)")
    print(timetotal)


if __name__ == "__main__": main()
