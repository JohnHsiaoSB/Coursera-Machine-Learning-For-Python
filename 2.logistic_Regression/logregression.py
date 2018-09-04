#!/usr/local/bin/python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from model.logrmodel import LogisticRegreesion_Model

def load_data():
    samples = np.loadtxt("ex2data1.txt",delimiter=",")

    X = samples[:,:2]
    y = samples[:,2]
    return (X,y)

def plotData(X,y, figObj):
    pos = X[np.where(y==1,True,False)]
    neg = X[np.where(y==0,True,False)]
    figObj.plot(pos[:,0],pos[:,1], "r+", markeredgewidth=1,markersize=6, label="Admitted")
    figObj.plot(neg[:,0],neg[:,1], "o", markerfacecolor='yellow', markersize=6, label="No admitted")

    figObj.set_xlabel("Exam 1 score")
    figObj.set_ylabel("Exam 2 score")
    figObj.legend(loc="upper right")

def main():
    parse = argparse.ArgumentParser(prog='logregression')
    parse.add_argument("--iterator", type=int, default=400, dest="iter")
    parse.add_argument("--use", default="python", dest="use", help="python or tensor")

    args_ret = parse.parse_args()
    (X_train, y_train) = load_data()

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    plotData(X_train, y_train, ax1)

    X = np.c_[np.ones((X_train.shape[0],1)) ,X_train]
    X = X.T #[3,100]=>[feature*3, sample*100]
    y = y_train.reshape(1,y_train.shape[0])

    model = LogisticRegreesion_Model(X, y, args_ret.use)
    model.iter = args_ret.iter

    cost, f_W = model.train()
    print("Learning result:")
    print("cost:"+str(cost))
    print("f_W:"+str(f_W))

    x_boundary = np.array([min(X.T[:,1]),max(X.T[:,1])])
    y_boundary = (-1/f_W[2])*(f_W[1]*x_boundary+f_W[0])

    plotData(X_train, y_train, ax2)
    ax2.plot(x_boundary,y_boundary,"b-", label='Decisoin Boundary')
    ax2.legend(loc="upper right");
    plt.show()

    print("45,85 predict:"+str(model.sigmoid(np.dot(f_W.T, np.array([[1],[45],[85]])))))

if __name__ == "__main__":
    main()
