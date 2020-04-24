#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:32:34 2019

@author: ashray

python 3.6 --  working great

rnn with mcmc in torch

"""

#  %reset
#  %reset -sf


import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import random
import math
import copy

# np.random.seed(1)

input_size = 320
hidden_size = 50
num_classes = 10
weightdecay = 0.01
batch_size = 10
device = 'cpu'

def data_load(data='train'):
    # trainsize = 200
    # testsize = 40

    if data == 'test':
        samples = torchvision.datasets.MNIST(root='./mnist', train=False, download=True,
                                             transform=torchvision.transforms.Compose([transforms.ToTensor(),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,))]))
        size = 20
        a, _ = torch.utils.data.random_split(samples, [size, len(samples) - size])

    else:
        samples = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=torchvision.transforms.Compose([transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
        size = 200
        a, _ = torch.utils.data.random_split(samples, [size, len(samples) - size])

    data_loader = torch.utils.data.DataLoader(a,
                                              batch_size=batch_size,
                                              shuffle=True)
    return data_loader

def f(): raise Exception("Found exit()")


class Model(nn.Module):

    # Defining input size, hidden layer size, output size and batch size respectively
    def __init__(self, topo, lrate):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(1024, 10)
        # self.fc2 = nn.Linear(128, 10)

        self.batch_size = batch_size
        self.sigmoid = nn.Sigmoid()
        self.topo = topo
        self.los = 0
        self.softmax = nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate)

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

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


class MCMC:
    def __init__(self, samples, topology, use_langevin_gradients, lr, batch_size=10):
        self.samples = samples
        self.topology = topology
        self.rnn = Model(topology, lr)
        self.traindata = data_load(data='train')
        self.testdata = data_load(data='test')
        self.topology = topology
        self.use_langevin_gradients = use_langevin_gradients
        self.batch_size = batch_size
        # ----------------

    def rmse(self, predictions, targets):
        return self.rnn.los.item()

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
            #images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = self.rnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total


    def sampler(self):
        samples = self.samples
        rnn = self.rnn
        w = rnn.state_dict()
        w_size = len(rnn.getparameters(w))
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)
        acc_train = np.zeros(samples)
        acc_test = np.zeros(samples)
        eta = 0
        w_proposal = np.random.randn(w_size)
        w_proposal = rnn.dictfromlist(w_proposal)
        step_w = 0.001
        train = self.traindata  # data_load(data='train')
        test = self.testdata  # data_load(data= 'test')
        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        delta_likelihood = 0.5  # an arbitrary position
        prior_current = self.prior_likelihood(sigma_squared, rnn.getparameters(w))


        [likelihood, pred_train, rmsetrain] = self.likelihood_func(rnn, train)
        [_, pred_test, rmsetest] = self.likelihood_func(rnn, test)

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

        num_accepted = 0
        langevin_count = 0
        init_count = 0
        rmse_train[0] = rmsetrain
        rmse_test[0] = rmsetest
        acc_train[0] = self.accuracy(train)
        acc_test[0] = self.accuracy(test)

        # acc_train[0] = 50.0
        # acc_test[0] = 50.0

        # print('i and samples')
        for i in range(samples):  # Begin sampling --------------------------------------------------------------------------

            lx = np.random.uniform(0, 1, 1)
            old_w = rnn.state_dict()

            if (self.use_langevin_gradients is True) and (lx < self.l_prob):
                w_gd = rnn.langevin_gradient(train)  # Eq 8
                w_proposal = rnn.addnoiseandcopy(0, step_w)  # np.random.normal(w_gd, step_w, w_size) # Eq 7
                w_prop_gd = rnn.langevin_gradient(train)
                wc_delta = (rnn.getparameters(w) - rnn.getparameters(w_prop_gd))
                wp_delta = (rnn.getparameters(w_proposal) - rnn.getparameters(w_gd))
                sigma_sq = step_w
                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq
                diff_prop = first - second
                diff_prop = diff_prop / self.adapttemp
                langevin_count = langevin_count + 1
            else:
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
            u = random.uniform(0, 1)

            if u < mh_prob:
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

            else:
                w = old_w
                rnn.loadparameters(w)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                print (i, rmsetrain, rmsetest, acc_train1, acc_test1, 'rejected')
                rmse_train[i,] = rmse_train[i - 1,]
                rmse_test[i,] = rmse_test[i - 1,]
                acc_train[i,] = acc_train[i - 1,]
                acc_test[i,] = acc_test[i - 1,]


        print ((num_accepted * 100 / (samples * 1.0)), '% was Accepted')

        print ((langevin_count * 100 / (samples * 1.0)), '% was Langevin')

        return acc_train, acc_test, rmse_train, rmse_test



def main():
    outres = open('resultspriors.txt', 'w')

    topology = [input_size, hidden_size, num_classes]

    numSamples = 500
    ulg = False

    learnr=0.001


    mcmc = MCMC(numSamples, topology, ulg, learnr)  # declare class
    acc_train, acc_test, rmse_train, rmse_test = mcmc.sampler()

    print("\n\n\n\n\n\n\n\n")
    print("Mean of RMSE Train")
    print(np.mean(rmse_train))
    print("\n")
    print("Mean of Accuracy Train")
    print(np.mean(acc_train))
    print("\n")
    print("Mean of RMSE Test")
    print(np.mean(rmse_test))
    print("\n")
    print("Mean of Accuracy Test")
    print(np.mean(acc_test))
    print ('sucessfully sampled')

if __name__ == "__main__": main()