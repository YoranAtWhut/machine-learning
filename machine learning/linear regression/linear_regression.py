# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

def linearRegression(alpha=0.01,num_iters=1400):
    print('loading data...\n')
    
    data = loadtxtAndcsv_data("data.txt",",",np.float64) #读取数据
    X = data[:,0:-1]
    y = data[:,-1]
    m = len(y)
    col = data.shape[1]
    print('col',col)
    
    X,mu,sigma = featureNormalization(X)
    plot_X1_X2(X)
    
    X = np.hstack((np.ones((m,1)),X)) #在X前加一列1
    #print(X)    
    
    print('executing gradient descent algorithm...\n')
    
    theta = np.zeros((col,1))
    y = y.reshape(-1,1) #将行向量转换为列向量
    theta,J_history = gradientDescent(X,y,theta,alpha,num_iters)
    
    plotJ(J_history,num_iters)
    
    return mu,sigma,theta,X,y

def loadtxtAndcsv_data(filename,split,dataType):
    return np.loadtxt(filename,delimiter=split,dtype=dataType)
    
def loadnpy_data(filename):
    return np.load(filename)
    
#归一化feature
def featureNormalization(X):
    X_norm = np.array(X)
    
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))
    
    mu = np.mean(X_norm,0) #求每一列平均值，0为行，1为列
    sigma = np.std(X_norm,0)
    for i in range(X.shape[1]):
        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]
    
    return X_norm,mu,sigma
    
def plot_X1_X2(X):
    plt.scatter(X[:,0],X[:,1])
    plt.show()
    
def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)
    n = len(theta)
    
    temp = np.matrix(np.zeros((n,num_iters))) #暂存每次迭代计算的theta,转换为矩阵形式
    
    J_history = np.zeros((num_iters,1))
    
    for i in range(num_iters):
        h = np.dot(X,theta)
        temp[:,i] = theta - ((alpha/m)*(np.dot(np.transpose(X),h-y)))
        theta = temp[:,i]
        J_history[i] = computeCost(X,y,theta)
        #print('.')
    return theta,J_history
    
def computeCost(X,y,theta):
    m = len(y)
    J = 0
    J = (np.transpose(X*theta-y))*(X*theta-y)/(2*m)
    return J

def plotJ(J_history,num_iters):
    x = np.arange(1,num_iters+1)
    plt.plot(x,J_history)
    plt.xlabel('iterations')
    plt.ylabel('value')
    plt.title('the J-variation during the iteration')
    plt.show()
    
def testLinearRegression():
    mu,sigma,theta,X,y = linearRegression(0.01,400)
    Intuitive(mu,sigma,theta,X,y)
    print('\n计算的theta值为：',theta)
    print('\n预测的结果为：%f'%predict(mu,sigma,theta))
    
def predict(mu,sigma,theta):
    result = 0
    
    predict = np.array([1650,3])
    norm_predict = (predict-mu)/sigma
    final_predict = np.hstack((np.ones((1)),norm_predict))
    
    result = np.dot(final_predict,theta)
    return result

def Intuitive(mu,sigma,theta,X,y):
    x = np.arange(1,len(y)+1)
    predict = np.dot(X,theta)
    plt.plot(x,predict)
    plt.plot(x,y)
    plt.show()
    

if __name__ == '__main__':
    testLinearRegression()
        
    