#!/usr/local/bin/python3
import numpy as np
import argparse
import matplotlib.pyplot as plt
from model.lrmodel import LinearR_Model

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = np.divide(X-mu,sigma)
    return X_norm, mu, sigma

def load_data():
    samples = np.loadtxt("ex1data2.txt",delimiter=",")
    X = samples[:,:2]
    y = samples[:,2]
    return (X,y)

def main():
    parse = argparse.ArgumentParser(prog='lregression_multi')
    parse.add_argument("--learn_rate", type=float, default=0.01, dest="lrate")
    parse.add_argument("--iterator", type=int, default=400, dest="iter")
    parse.add_argument("--use", default="python", dest="use", help="python or tensor")

    args_ret = parse.parse_args()
    (X_train, y_train) = load_data()

    X_norm, mu, sigma = featureNormalize(X_train)
    #X shape transpose to [features, samples] [2,97]
    X_norm = X_norm.T
    #y shape reshape to [1, samples]=>[1,97]
    y = y_train.reshape(1,y_train.shape[0])


    model = LinearR_Model(X_norm, y, args_ret.use)

    model.lrate = args_ret.lrate
    model.iter = args_ret.iter

    costs, W, b = model.train()
    print("Learning result:")
    print("W:"+str(W))
    print("b:"+str(b))
    plt.plot(costs)
    plt.xlabel('Iterator')
    plt.ylabel('cost')
    plt.title('learing rate '+str(args_ret.lrate))
    plt.show()

    p_val = np.array([[1650, 3]])
    p_val = np.divide(p_val-mu, sigma)
    print("predict price:"+str(np.sum(np.dot(p_val,W)+b)))

if __name__ == "__main__":
    main()
