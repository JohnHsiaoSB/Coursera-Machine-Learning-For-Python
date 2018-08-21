import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

class LinearR_Model:
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

    def computeCost(self,W, b):
        m = self.X_train.shape[1]
        z = np.dot(W.T, self.X_train)+b
        J = z-self.Y_train
        cost = np.sum(np.square(J))/(2*m)
        return cost, J

    def optimizeGradient(self,W, b):
        cost, J = self.computeCost(W,b)
        m = self.X_train.shape[1]
        dw = 1/m*np.dot(J,self.X_train.T).T
        db = 1/m*np.sum(J, axis=1, keepdims=1)

        grads = {
            'dw': dw,
            'db': db
        }
        return cost, grads
    def train_for_tensor(self):
        X = tf.placeholder(tf.float32,shape=[self.X_train.shape[0],None], name="X_train")
        Y = tf.placeholder(tf.float32,shape=[self.Y_train.shape[0],None], name="Y_train")

        W = tf.get_variable("W", shape=[self.X_train.shape[0],1], initializer=tf.zeros_initializer())
        b = tf.get_variable("b", shape=[1,1], initializer=tf.zeros_initializer())
        z = tf.add(tf.matmul(W, X),b)
        cost = tf.reduce_mean(tf.square(z-self.Y_train))

        train = tf.train.GradientDescentOptimizer(self.lrate).minimize(cost)
        costs = []
        final_W=0
        final_b=0
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for it in range(self.iter):
                _, cost_result = sess.run([train, cost], feed_dict={X:self.X_train, Y:self.Y_train})
                costs.append(cost_result)
                if it % 10 == 0:
                    print("cost:"+str(cost_result))
            final_W = sess.run(W)
            final_b = sess.run(b)
        return costs, final_W, final_b

    def trains_for_python(self):
        W = np.zeros((self.X_train.shape[0],1))
        b = np.zeros((1,1))
        costs = []

        for i in range(self.iter):
            cost, grads = self.optimizeGradient(W, b)
            costs.append(cost)

            dw = grads['dw']
            db = grads['db']

            W = W - self.lrate*dw
            b = b - self.lrate*db
            #if i%10 == 0:
            #    print("W:"+str(W))
            #    print("b:"+str(b))
            #    print("cost:"+str(cost))

        return costs, W,b

    def train(self):
        if self.use == 'tensor':
            return self.train_for_tensor()
        else:
            return self.trains_for_python()
