import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from random import shuffle
import matplotlib.pyplot as plt
import random
import time
import math
import numpy as np
from torchvision import datasets, transforms
import mcmcplot as mcmcplt
import os
import copy
from torchsummary import summary
import torch.optim as optim


class NN(nn.Module):

    def __init__(self, topo, lrate, x, y):
        super(NN, self).__init__()
        # Defining input size, hidden layer size, output size and batch size respectively
        self.fsize=topo[0]
        self.ch1=topo[1]
        self.ch2=topo[3]
        self.topo=topo
        self.n_in, self.n_h, self.n_out, self.batch_size = topo[4], topo[5], topo[6], 1
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        self.learnRate = lrate
        self.F1=np.random.randn(1, self.topo[0]*self.topo[0]*self.topo[1])
        self.F2=np.random.randn(1, self.topo[2]*self.topo[2]*self.topo[3]*self.topo[1])
        self.FB1=np.random.randn(1, self.topo[1])
        self.FB2=np.random.randn(1, self.topo[3])
        self.W1=np.random.randn(1, self.topo[4]*self.topo[5])
        self.W2=np.random.randn(1, self.topo[5]*self.topo[6])
        self.B1=np.random.randn(1, self.topo[5])
        self.B2=np.random.randn(1, self.topo[6])
        print('Creating CNN Model')
        # Create a model
        self.conv1 = nn.Conv2d(1,self.ch1,self.fsize,1)
        self.conv2 = nn.Conv2d(self.ch1,self.ch2,self.fsize,1)
        self.fc1 = nn.Linear(self.n_in,self.n_h)
        self.fc2 = nn.Linear(self.n_h,self.n_out)


    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=0)
        #output,index = torch.max(output,1)
        return output



    def evaluate_proposal(self, x, w=None):
        x = torch.FloatTensor(x)
        #print(x.shape)
        if w is None:
            y_pred = self.forward(x)
            return copy.deepcopy(y_pred.detach().numpy())
        else:
            d = copy.deepcopy(self.state_dict())
            self.loadparameters(w)
            y_pred = self.forward(x)
            self.loadparameters(d)
            return copy.deepcopy(y_pred.detach().numpy())

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
    def addnoiseandcopy(self, w, mea, std_dev):
        dic = {}
        for name in (w.keys()):
            dic[name] = copy.deepcopy(w[name]) + torch.zeros(w[name].size()).normal_(mean=mea, std=std_dev)
        return dic


class MCMC:
    def __init__(self, samples, learnrate, train_x, train_y,test_x,test_y, topology):
        self.samples = samples  # max epocs
        self.topology = topology  # NN topology [input, hidden, output]
        self.train_x = train_x#
        self.test_x = test_x
        self.train_y=train_y
        self.test_y=test_y
        self.learnrate = learnrate
        # ----------------

    def rmse(self, predictions, targets):
        predictions = (predictions)
        targets=np.array(targets)
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, x,y, w, tausq):
        #y = data[:, self.topology[0]]
        y=y
        fx = neuralnet.evaluate_proposal(x, w)
        fx = torch.from_numpy(fx)
        fx=torch.argmax(fx,1)
        fx=fx.detach().numpy()
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(np.array(y) - np.array(fx)) / tausq
        return [np.sum(loss), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w_list, tausq):
        h = self.topology[5]  # number hidden neurons
        d = self.topology[4]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w_list)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def sampler(self):

        # ------------------- initialize MCMC
        testsize = self.test_x.shape[0]  # self.testdata.shape[0]
        trainsize = self.train_x.shape[0]
        samples = self.samples

        netw = self.topology  # [input, hidden, output]

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        y_test = self.test_y  # self.testdata[:, netw[0]]
        y_train = self.train_y  # self.traindata[:, netw[0]]


        # print(len(y_train))
        # print(len(y_test))

        # here
        #w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias
        #w_size=w_size+(3*3*32)+(3*3*64)+32+64
        #print(w_size)
        w_size=225034

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        # original -->    fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        # print('shape: ',np.array(y_train).shape[1])
        #fxtrain_samples = np.ones(
            #(samples, trainsize, int(np.array(y_train).shape[1])))  # fx of train data over all samples

        fxtrain_samples = np.ones((samples, trainsize))


        # original --> fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples || probably for 1 dimensional data

        #fxtest_samples = np.ones(
         #   (samples, testsize, np.array(self.test_y).shape[1]))  # fx of test data over all samples

        fxtest_samples = np.ones((samples, testsize))

        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)

        '''
        weight already declared when NN was created with topology defined in it
        '''
        step_w = 0.02  # defines how much variation you need in changes to w
        step_eta = 0.01
        # --------------------- Declare FNN and initialize

        neuralnet = NN(self.topology, self.learnrate, self.train_x, self.train_y)

        #summary(neuralnet,(1,28,28))

        #x = neuralnet.encode()

        #neuralnet.train()

        #for name, param in neuralnet.named_parameters():
         #   if param.requires_grad:
         #       print(name, param.data)

        optimizer = optim.SGD(neuralnet.parameters(), lr=0.05)
        loss_fn = nn.MSELoss()

        w = copy.deepcopy(neuralnet.state_dict())

        print('Evaluating Initial w')

        '''
        pred1 = neuralnet.evaluate_proposal(self.train_x[:3])
        print(neuralnet.model.state_dict())
        d = neuralnet.addnoiseandcopy(0,1)
        print(d)
        neuralnet.loadparameters(d)
        print(neuralnet.model.state_dict())
        pred2 = neuralnet.evaluate_proposal(self.train_x[:3])
        print(pred1,'\n',pred2)
        return
        '''

        # print(w,np.array(self.train_x).shape)
        pred_train = neuralnet.evaluate_proposal(self.train_x, w)
        pred_test = neuralnet.evaluate_proposal(self.test_x, w)

        pred_train = torch.from_numpy(pred_train)
        pred_train=torch.argmax(pred_train,1)
        pred_train=pred_train.detach().numpy()

        #print(pred_train.shape)



        eta = np.log(np.var((pred_train) - np.array(y_train)))




        tau_pro = np.exp(eta)

        #        err_nn = np.sum(np.square((pred_train) - np.array(y_train)))/(len(pred_train)) #added by ashray mean square sum
        #        print('err_nn is: ',err_nn)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        # print(pred_train)
        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, neuralnet.getparameters(w),
                                                 tau_pro)  # takes care of the gradients
        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.train_x, self.train_y, w,
                                                                   tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.test_x, self.test_y, w,
                                                                        tau_pro)

        # print(likelihood,' is likelihood of train')
        # print(pred_train)
        # print(pred_train, ' is pred_train')




        naccept = 0

        print('Sampling using Langevin Gradient MCMC')
        plt.plot(x_train, y_train)
        plt.plot(x_train, pred_train)
        plt.title("Plot of Data vs Initial Fx")
        if os.path.exists('mcmcresults') is False:
            os.makedits('mcmcresults')
        plt.savefig('mcmcresults/begin.png')
        plt.clf()

        plt.plot(x_train, y_train)

        for i in range(samples - 1):
            # print(i)

            # w_proposal = w + np.random.normal(0, step_w, w_size)

            optimizer.zero_grad()
            predictions=neuralnet.forward(self.train_x)

            #predictions = torch.from_numpy(predictions)
            #predictions = torch.argmax(predictions, 1)
            #predictions = predictions.detach().numpy()

            #print(predictions.shape)
            #print(self.train_y.shape)

            #predictions = predictions.to(dtype=torch.long)
            train_y1 = self.train_y.to(dtype=torch.long)
            #loss = loss_fn(predictions, self.train_y)
            loss=F.nll_loss(predictions, train_y1)
            loss.backward()
            optimizer.step()
            w=neuralnet.state_dict()

            w_proposal = neuralnet.addnoiseandcopy(w, 0,
                                                   step_w)  # adding gaussian normal distributed noise to all weights and all biases

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.train_x,
                                                                                self.train_y, w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.test_x, self.test_y,
                                                                            w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, neuralnet.getparameters(w_proposal),
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood

            # mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))
            mh_prob = min(0, (diff_likelihood + diff_priorliklihood))
            mh_prob = math.exp(mh_prob)
            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                print(i, ' is the accepted sample')
                naccept += 1
                likelihood = likelihood_proposal
                prior_likelihood = prior_prop
                w = w_proposal
                neuralnet.loadparameters(copy.deepcopy(w_proposal))
                # w = neuralnet.getparameters(w = w_proposal)
                eta = eta_pro
                # if i % 100 == 0:
                #     #print ( likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')
                #     print ('Sample:',i, 'RMSE train:', rmsetrain, 'RMSE test:',rmsetest)

                # pos_w[i + 1,] = w_proposal
               # print(neuralnet.getparameters(w_proposal).reshape(-1).shape)
                #pos_w[i + 1,] = copy.deepcopy(neuralnet.getparameters(w_proposal)).reshape(-1)
                pos_tau[i + 1,] = tau_pro
                fxtrain_samples[i + 1,] = pred_train
                fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest

                plt.plot(x_train, pred_train)


            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]

                # print i, 'rejected and retained'

            if i % 100 == 0:
                # print ( likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')
                print('Sample:', i, 'RMSE train:', rmsetrain, 'RMSE test:', rmsetest)

        print(naccept, ' num accepted')
        print((naccept * 100) / (samples * 1.0), '% was accepted')
        accept_ratio = naccept / (samples * 1.0) * 100

        plt.title("Plot of Accepted Proposals")
        plt.savefig('mcmcresults/proposals.png')
        plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
        plt.clf()

        return (
        pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)


def main():
    outres = open('resultspriors.txt', 'w')
    learnRate = 0.1

    # for mackey
    #fname = "train_mackey.txt"
    #x, y = data_loader(fname)
    # print_data(x,y)
    #x, y = shuffledata(x, y)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=60000, shuffle=True)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    images = images.to(dtype=torch.float32)
    labels = labels.to(dtype=torch.float32)

    #test_loader = torch.utils.data.DataLoader(
     #   datasets.MNIST('../data', train=False, transform=transforms.Compose([
      #      transforms.ToTensor(),
       #     transforms.Normalize((0.1307,), (0.3081,))
        #])),
        #batch_size=60000, shuffle=True)

    #tr = train_loader.dataset.data
    #tr = tr.to(dtype=torch.float32)
    #tr=tr.unsqueeze(0)

    train_x = images[0:10000]
    test_x = images[50000:60000]

    #tr1 = train_loader.dataset.targets
    #tr1 = tr1.to(dtype=torch.float32)
    #tr1=tr1.unsqueeze(0)

    train_y = labels[0:10000]
    test_y = labels[50000:60000]



    #train_x = x[:int(len(x) * 0.8)]
    #test_x = x[int(len(x) * 0.8):]
    #train_y = y[:int(len(y) * 0.8)]
    #test_y = y[int(len(y) * 0.8):]
    # Input = len(train_x[0][0])
    # Output = len(train_y[0])
    FilterSize=3
    Chanel1=32
    Chanel2=64
    Input = 1600
    Output = 10
    # print(traindata)
    Hidden = 128
    topology = [FilterSize, Chanel1, FilterSize, Chanel2, Input, Hidden, Output]
    numSamples = 1000  # need to decide yourself

    mcmc = MCMC(numSamples, learnRate, train_x, train_y, test_x, test_y, topology)  # declare class

    [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler()
    print('sucessfully sampled')

    burnin = 0.1 * numSamples  # use post burn in samples

    pos_w = pos_w[int(burnin):, ]
    pos_tau = pos_tau[int(burnin):, ]

    '''
    to plots the histograms of weight destribution
    '''

    #mplt.initialiseweights(len(pos_w), len(pos_w[0]))
    #for i in range(len(pos_w)):
     #   mplt.addweightdata(i, pos_w[i])
    #mplt.saveplots()

    fx_mu = fx_test.mean(axis=0)
    fx_high = np.percentile(fx_test, 95, axis=0)
    fx_low = np.percentile(fx_test, 5, axis=0)

    fx_mu_tr = fx_train.mean(axis=0)
    fx_high_tr = np.percentile(fx_train, 95, axis=0)
    fx_low_tr = np.percentile(fx_train, 5, axis=0)

    rmse_tr = np.mean(rmse_train[int(burnin):])
    rmsetr_std = np.std(rmse_train[int(burnin):])
    rmse_tes = np.mean(rmse_test[int(burnin):])
    rmsetest_std = np.std(rmse_test[int(burnin):])
    print(rmse_tr, rmsetr_std, rmse_tes, rmsetest_std)
    np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')

    # ytestdata = testdata[:, input]
    # ytraindata = traindata[:, input]
    ytestdata = test_y
    ytraindata = train_y

    # converting everything to np arrays
    x_test = np.array(x_test)
    fx_low = np.array(fx_low)
    fx_high = np.array(fx_high)
    fx_mu = np.array(fx_mu)
    ytestdata = np.array(ytestdata)
    x_train = np.array(x_train)
    ytraindata = np.array(ytraindata)
    fx_mu_tr = np.array(fx_mu_tr)
    fx_low_tr = np.array(fx_low_tr)
    fx_high_tr = np.array(fx_high_tr)

    plt.plot(x_test, ytestdata, label='actual')
    plt.plot(x_test, fx_mu, label='pred. (mean)')
    plt.plot(x_test, fx_low, label='pred.(5th percen.)')
    plt.plot(x_test, fx_high, label='pred.(95th percen.)')
    # print(np.array(x_test).shape,np.array(fx_low).shape,np.array(fx_high).shape)
    # print(fx_low[:,0],fx_high,x_test)
    plt.fill_between(x_test, fx_low[:, 0], fx_high[:, 0], facecolor='g', alpha=0.4)
    plt.legend(loc='upper right')

    plt.title("Plot of Test Data vs MCMC Uncertainty ")
    plt.savefig('mcmcresults/mcmcrestest.png')
    plt.savefig('mcmcresults/mcmcrestest.svg', format='svg', dpi=600)
    plt.clf()
    # -----------------------------------------
    plt.plot(x_train, ytraindata, label='actual')
    plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
    plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
    plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
    plt.fill_between(x_train, fx_low_tr[:, 0], fx_high_tr[:, 0], facecolor='g', alpha=0.4)
    plt.legend(loc='upper right')

    plt.title("Plot of Train Data vs MCMC Uncertainty ")
    plt.savefig('mcmcresults/mcmcrestrain.png')
    plt.savefig('mcmcresults/mcmcrestrain.svg', format='svg', dpi=600)
    plt.clf()

    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)

    ax.boxplot(pos_w)

    ax.set_xlabel('[W1] [B1] [W2] [B2]')
    ax.set_ylabel('Posterior')

    plt.legend(loc='upper right')

    plt.title("Boxplot of Posterior W (weights and biases)")
    plt.savefig('mcmcresults/w_pos.png')
    plt.savefig('mcmcresults/w_pos.svg', format='svg', dpi=600)

    plt.clf()


if __name__ == "__main__": main()