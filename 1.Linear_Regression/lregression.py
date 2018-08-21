#!/usr/local/bin/python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from model.lrmodel import LinearR_Model

def load_data():
    samples = np.loadtxt("ex1data1.txt",delimiter=",")

    X = samples[:,0]
    y = samples[:,1]

    X = X.reshape(1,X.shape[0])
    y = y.reshape(1,y.shape[0])

    return (X,y)


def main():
    parse = argparse.ArgumentParser(prog='lregression')
    parse.add_argument("--learn_rate", type=float, default=0.01, dest="lrate")
    parse.add_argument("--iterator", type=int, default=1500, dest="iter")
    parse.add_argument("--use", default="python", dest="use", help="python or tensor")

    args_ret = parse.parse_args()
    (X_train, y_train) = load_data()
    model = LinearR_Model(X_train, y_train, args_ret.use)
    model.lrate = args_ret.lrate
    model.iter = args_ret.iter
    costs, W, b = model.train()
    print("Learning result:")
    print("W:"+str(W))
    print("b:"+str(b))
    y_h = np.dot(W.T, X_train)+b

    #Plot figure
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3, projection='3d')

    ax1.scatter(X_train,y_train, s=30, c='r', marker='x',linewidths=1)
    ax1.plot(X_train.reshape(X_train.shape[1]),y_h.reshape(y_h.shape[1]), 'b--', marker='x', label='Linear Regression')
    ax1.legend(loc="upper left");

    ax2.plot(costs)
    ax2.set_xlabel('Iterator')
    ax2.set_ylabel('cost')
    ax2.set_title('learing rate '+str(args_ret.lrate))

    t0 = np.linspace(-10,10,50)
    t1 = np.linspace(-1,4,50)
    th0,th1 = np.meshgrid(t0,t1, indexing='xy')
    P = np.zeros((t0.size, t1.size))

    for (i,j),v in np.ndenumerate(P):
        P[i,j],_ = model.computeCost(np.array([[th1[i,j]]]), np.array([[th0[i,j]]]))
    ax3.plot_surface(th0,th1, P, rstride=1,cstride=1, alpha=0.6, cmap=plt.cm.jet)
    ax3.set_xlabel(r'$\theta_0$',fontsize=17)
    ax3.set_ylabel(r'$\theta_1$',fontsize=17)
    ax3.set_zlabel('Cost')
    ax3.set_zlim(P.min(),P.max())
    ax3.view_init(elev=15, azim=230)
    plt.show()

    print("show learning result:")
    print("W:"+str(W))
    print("b:"+str(b))

if __name__ == "__main__":
    main()
