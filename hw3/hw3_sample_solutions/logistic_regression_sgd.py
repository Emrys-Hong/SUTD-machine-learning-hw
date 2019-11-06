'''Logistic Regression with Stocastic Gradient Descent'''
import numpy as np
import random
import matplotlib.pyplot as plt

#Stochastic Gradient Descent
def SGD(trainMat):
    MaxIter = 10000
    learning_rate = 0.1
    LogRSGD_StoreInterval = 100
    
    [N,d] = trainMat.shape
    w = np.zeros((1,d))
    delta_w = w
    trainD = np.concatenate((np.ones((N,1)),trainMat[:,1:]), axis = 1)
    assert trainD.shape[1] == d
    w = np.zeros((1,d))
    delta_w = np.zeros((1,d))
    trainLL_list = []
    
    for t in range(MaxIter):
        i = random.randint(0,N-1)
        delta_w = -trainMat[i,0]*trainD[i,:]/(1+np.exp(trainMat[i,0]*np.inner(trainD[i,:],w)))
        w = w - learning_rate* delta_w
        if t%LogRSGD_StoreInterval == 0:
            print ('iteration times: ', t)
            trainLL = 0
            for i in range(N):
                trainLL += np.log(1+ np.exp(-trainMat[i,0]*np.inner(trainD[i,:],w)))
            trainLL = -trainLL
            trainLL_list.append(trainLL[0]/N)

    (noop,llen) = w.shape
    for i in range(llen):
        print ('w[%d] = %f' % (i,w[0][i]))
        
    T = len(trainLL_list)
    x = range(0,T)
    for i in range(len(x)):
        x[i] = LogRSGD_StoreInterval*x[i]
    plt.plot(x, trainLL_list, 'b')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Log-likelihood")
    plt.show()

    return w

def main():
    trainMat = np.loadtxt('train_diabetes.csv', delimiter=',')

    w = SGD(trainMat)
    
    print (w)



if __name__=="__main__":
    main()
