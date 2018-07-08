import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

def test():
    mat = loadmat("svm\data\ex6data1.mat")
    X = mat["X"]
    y = mat["y"]
    models = [svm.SVC(C, kernel='linear') for C in [1, 100]]
    clfs = [model.fit(X, y) for model in models]
    title = ['SVM Decision Boundary with C = {} (Example Dataset 1'.format(C) for C in [1, 100]]
    for model, title in zip(clfs, title):
        print(str(model.predict(X)))
        plt.figure(figsize=(8, 5))
        plotData(X, y)
        plotBoundary(model, X)
        plt.title(title)
    plt.show()

def test2():
    mat = loadmat("svm\data\ex6data2.mat")
    X = mat["X"]
    y = mat["y"]
    gamma = np.power(0.1,-2.)/2
    models = [svm.SVC(C, kernel='rbf',gamma=gamma) for C in [1, 100]]
    clfs = [model.fit(X, y) for model in models]
    title = ['SVM Decision Boundary with C = {} (Example Dataset 1'.format(C) for C in [1, 100]]
    for model, title in zip(clfs, title):
        plt.figure(figsize=(8, 5))
        plotData(X, y)
        plotBoundary(model, X)
        plt.title(title)
    plt.show()

def test3():
    mat = loadmat("svm\datawaa\ex6data3.mat")
    X = mat["X"]
    y = mat["y"]
    Xval = mat["Xval"]
    yval = mat["yval"]
    Cvalues = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
    sigmavalues = Cvalues
    best_pair, best_score = (0, 0), 0
    best_model = None

    for C in Cvalues:
        for sigma in sigmavalues:
            gamma = np.power(0.1, -2.) / 2
            model = svm.SVC(C, kernel='rbf', gamma=gamma)
            clfs = [model.fit(X,y.flatten())]
            this_score = model.score(Xval,yval)
            if this_score > best_score:
                best_score = this_score
                best_pair = (C,sigma)
                best_model = model

    print('best_pair={}, best_score={}'.format(best_pair, best_score))
    plt.figure(figsize=(8, 5))
    plotData(X, y)
    plotBoundary(best_model, X)
    plt.show()

def plotData(X,y):
    # plt.figure(figsize=(8,5))
    plt.scatter(X[:,0],X[:,1],c=y.flatten(),cmap="rainbow")
    # print(str(X[:,0]))
    # print(str(y.flatten()))
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()

def plotBoundary(clf,X):
    x_min, x_max = X[:, 0].min() * 1.2, X[:, 0].max() * 1.1
    y_min, y_max = X[:, 1].min() * 1.1, X[:, 1].max() * 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)

def gaussKernel(x1,x2,sigma):
    return np.exp(-np.power((x1-x2),2).sum()/(2*np.power(sigma,2)))

if __name__ == '__main__':
    test()