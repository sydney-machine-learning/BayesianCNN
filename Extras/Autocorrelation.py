import emcee.autocorr as mc
import numpy as np
import matplotlib.pyplot as plt

weight0=np.zeros((10,200))
weight100=np.zeros((10,200))
weight40000=np.zeros((10,200))
weight50000=np.zeros((10,200))
weight60000=np.zeros((10,200))

temp = [1.0, 1.080059738892306, 1.360790000174377, 1.851749424574581, 1.1665290395761165, 1.2599210498948732, 1.4697344922755988, 1.5874010519681994, 1.7144879657061458, 2.0]


for i in range(10):

    file_name = 'weight[0]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    weight0[i,:]=dat

    file_name = 'weight[100]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    weight100[i,:]=dat

    file_name = 'weight[40000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    weight40000[i,:]=dat

    file_name = 'weight[50000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    weight50000[i,:]=dat

    file_name = 'weight[60000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    weight60000[i,:]=dat

    x1 = np.linspace(0, 200, num=200)

    plt.plot(x1, weight0[i], label='Weight[0]')
    plt.legend(loc='upper right')
    plt.title("Weight[0]_Chain"+str(temp[i])+ " Trace")
    plt.savefig('singlechain/weight[0]_Chain' + str(temp[i])+'_samples.png')
    plt.clf()

    plt.hist(weight0[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig('singlechain/weight[0]_Chain' +str(temp[i])+'_hist.png')
    plt.clf()

    plt.plot(x1, weight100[i], label='Weight[100]')
    plt.legend(loc='upper right')
    plt.title("Weight[100]_Chain" + str(temp[i]) + " Trace")
    plt.savefig('singlechain/weight[100]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight100[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig('singlechain/weight[100]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.plot(x1, weight40000[i], label='Weight[40000]')
    plt.legend(loc='upper right')
    plt.title("Weight[40000]_Chain" + str(temp[i]) + " Trace")
    plt.savefig('singlechain/weight[40000]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight40000[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig('singlechain/weight[40000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.plot(x1, weight50000[i], label='Weight[50000]')
    plt.legend(loc='upper right')
    plt.title("Weight[50000]_Chain" + str(temp[i]) + " Trace")
    plt.savefig('singlechain/weight[50000]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight50000[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig('singlechain/weight[50000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.plot(x1, weight60000[i], label='Weight[60000]')
    plt.legend(loc='upper right')
    plt.title("Weight[60000]_Chain" + str(temp[i]) + " Trace")
    plt.savefig('singlechain/weight[60000]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight60000[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig('singlechain/weight[60000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()




weight0 = weight0.reshape((10*200), 1)
weight100 = weight100.reshape((10*200), 1)
weight40000 = weight40000.reshape((10*200), 1)
weight50000 = weight50000.reshape((10*200), 1)
weight60000 = weight60000.reshape((10*200), 1)



iact0 = mc.integrated_time(weight0, quiet=True)
iact100 = mc.integrated_time(weight100, quiet=True)
iact40000 = mc.integrated_time(weight40000, quiet=True)
iact50000 = mc.integrated_time(weight50000, quiet=True)
iact60000 = mc.integrated_time(weight60000, quiet=True)

print(iact0)
print('\n')

print(iact100)
print('\n')

print(iact40000)
print('\n')

print(iact50000)
print('\n')

print(iact60000)
print('\n')


