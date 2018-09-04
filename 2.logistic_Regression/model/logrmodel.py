import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from tensorflow.python.framework import ops

class LogisticRegreesion_Model:
    def __init__(self, X, Y,use='python'):
        self.X_train = X
        self.Y_train = Y
        self.use = use
        self.iter = 1500
        self.lrate = 0.01
    def iter(self):
        return self.iter
    def lrate(self):
        return self.lrate
    def iter(self, v):
        self.iter = v
    def lrate(self,v):
        self.lrate = v

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def computeCost(self,W):
        m = self.X_train.shape[1]
        hx = self.sigmoid(np.dot(W.T, self.X_train))
        J = (-1/m)*(np.dot(self.Y_train,np.log(hx).T)+np.dot(1-self.Y_train,np.log(1-hx).T))
        return J

    def gradient(self,W):
        m = self.X_train.shape[1]
        hx = self.sigmoid(np.dot(W.T, self.X_train))
        dw = 1/m*np.dot(hx-self.Y_train, self.X_train.T).T
        #must flatten for minimize function
        return dw.flatten()

    def train_for_tensor(self):
        pass

    def trains_for_python(self):
        W = np.zeros((self.X_train.shape[0],1))
        result = minimize(self.computeCost, W, method=None, jac=self.gradient, options={"maxiter":self.iter})

        #result.fun is final cost function
        #result.x are theta parameters
        return result.fun, result.x.reshape(result.x.shape[0],1)

    def train(self):
        if self.use == 'tensor':
            return self.train_for_tensor()
        else:
            return self.trains_for_python()
