import numpy as np
import matplotlib.pyplot as plt

weight0=np.zeros((10,1000))
weight100=np.zeros((10,1000))
weight1000=np.zeros((10,1000))
weight4000=np.zeros((10,1000))
weight8000=np.zeros((10,1000))

temp = [1.0, 1.080059738892306, 1.360790000174377, 1.851749424574581, 1.1665290395761165, 1.2599210498948732, 1.4697344922755988, 1.5874010519681994, 1.7144879657061458, 2.0]


for i in range(10):

    file_name = 'TimeSeries/predictions/weight[0]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    weight0[i, :] = dat

    file_name = 'TimeSeries/predictions/weight[100]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    weight100[i, :] = dat

    file_name = 'TimeSeries/predictions/weight[1000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    weight1000[i, :] = dat

    file_name = 'TimeSeries/predictions/weight[4000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    weight4000[i, :] = dat

    file_name = 'TimeSeries/predictions/weight[8000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    weight8000[i, :] = dat

    x1 = np.linspace(0, 1000, num=1000)

    plt.plot(x1, weight0[i], label='Weight[0]')
    plt.legend(loc='upper right')
    plt.title("Weight[0]_Chain"+str(temp[i]) + " Trace")
    plt.savefig(
        'TimeSeries/autocorelation/weight[0]_Chain' + str(temp[i])+'_samples.png')
    plt.clf()

    plt.hist(weight0[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'TimeSeries/autocorelation/weight[0]_Chain' + str(temp[i])+'_hist.png')
    plt.clf()

    plt.plot(x1, weight100[i], label='Weight[100]')
    plt.legend(loc='upper right')
    plt.title("Weight[100]_Chain" + str(temp[i]) + " Trace")
    plt.savefig(
        'TimeSeries/autocorelation/weight[100]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight100[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'TimeSeries/autocorelation/weight[100]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.plot(x1, weight1000[i], label='Weight[1000]')
    plt.legend(loc='upper right')
    plt.title("Weight[1000]_Chain" + str(temp[i]) + " Trace")
    plt.savefig(
        'TimeSeries/autocorelation/weight[1000]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight1000[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'TimeSeries/autocorelation/weight[1000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.plot(x1, weight4000[i], label='Weight[4000]')
    plt.legend(loc='upper right')
    plt.title("Weight[4000]_Chain" + str(temp[i]) + " Trace")
    plt.savefig(
        'TimeSeries/autocorelation/weight[4000]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight4000[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'TimeSeries/autocorelation/weight[4000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.plot(x1, weight8000[i], label='Weight[8000]')
    plt.legend(loc='upper right')
    plt.title("Weight[8000]_Chain" + str(temp[i]) + " Trace")
    plt.savefig(
        'TimeSeries/autocorelation/weight[8000]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight8000[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'TimeSeries/autocorelation/weight[8000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()



weight0 = weight0.T
weight100 = weight100.T
weight1000 = weight1000.T
weight4000 = weight4000.T
weight8000 = weight8000.T

data = np.stack((weight0,weight100,weight1000,weight4000,weight8000), axis=2)

Nchains, Nsamples, Npars = data.shape

B_on_n = data.mean(axis=1).var(axis=0)      # variance of in-chain means
W = data.var(axis=1).mean(axis=0)           # mean of in-chain variances

#print(B_on_n, ' B_on_n mean')

#print(W, ' W variance ')

# simple version, as in Obsidian
sig2 = (Nsamples/(Nsamples-1))*W + B_on_n
Vhat = sig2 + B_on_n/Nchains
Rhat = Vhat/W

print(Rhat, ' Rhat')


# advanced version that accounts for ndof
m, n = np.float(Nchains), np.float(Nsamples)
si2 = data.var(axis=1)
xi_bar = data.mean(axis=1)
xi2_bar = data.mean(axis=1)**2
var_si2 = data.var(axis=1).var(axis=0)
allmean = data.mean(axis=1).mean(axis=0)
cov_term1 = np.array([np.cov(si2[:,i], xi2_bar[:,i])[0,1]
                        for i in range(Npars)])
cov_term2 = np.array([-2*allmean[i]*(np.cov(si2[:,i], xi_bar[:,i])[0,1])
                        for i in range(Npars)])
var_Vhat = ( ((n-1)/n)**2 * 1.0/m * var_si2
            +   ((m+1)/m)**2 * 2.0/(m-1) * B_on_n**2
            +   2.0*(m+1)*(n-1)/(m*n**2)
                * n/m * (cov_term1 + cov_term2))
df = 2*Vhat**2 / var_Vhat

#print(df, ' df ')
#print(var_Vhat, ' var_Vhat')
#print "gelman_rubin(): var_Vhat = {}, df = {}".format(var_Vhat, df)


Rhat *= df/(df-2)

print(Rhat, ' Rhat Advanced')
