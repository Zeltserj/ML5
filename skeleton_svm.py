#################################
# Your name: Jonathan Zeltser
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def init_classifiers(C, X_train, y_train):

    linear = svm.SVC(C, kernel='linear')
    linear.fit(X_train, y_train)

    quad = svm.SVC(C, kernel='poly', degree=2)
    quad.fit(X_train, y_train)

    rbf = svm.SVC(C, kernel='rbf')
    rbf.fit(X_train, y_train)
    return linear, quad, rbf


def create_three_plots(X_train, y_train,  linear, quad, rbf):

    names = ['Linear', 'Quadratic', 'RBF']
    clfs = [linear, quad, rbf]
    for i, name in enumerate(names):
        create_plot(X_train, y_train, clfs[i])
        plt.show()


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    C = 1000
    linear, quad, rbf = init_classifiers(C, X_train, y_train)
    create_three_plots(X_train, y_train, linear, quad, rbf)
    return np.vstack([linear.n_support_, quad.n_support_, rbf.n_support_])

def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    accuracies = np.empty(11)
    Cs = [10**i for i in range(-5, 6)]
    best_C = 0
    max_acc = 0
    for i, C in enumerate(Cs):
        linear = svm.SVC(C, kernel='linear')
        linear.fit(X_train, y_train)
        accuracy = linear.score(X_val, y_val)
        if accuracy > max_acc:
            max_acc = accuracy
            best_C = C
        accuracies[i] = accuracy
        create_plot(X_val,y_val,linear)
        plt.show()
    # plt.plot(Cs, accuracies, color='blue')
    # plt.scatter(best_C, max_acc, color='red')
    # plt.title('Accuracy as function of C')
    # plt.ylabel('Accuracy')
    # plt.xlabel('C')
    # plt.xscale('log')
    plt.show()
    return accuracies

def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    accuracies = np.empty(11)
    gammas = [10 ** i for i in range(-5, 6)]
    best_gamma = 0
    max_acc = 0
    for i, gamma in enumerate(gammas):
        rbf = svm.SVC(C=10, kernel='rbf', gamma=gamma)
        rbf.fit(X_train, y_train)
        accuracy = rbf.score(X_val, y_val)
        if accuracy > max_acc:
            max_acc = accuracy
            best_gamma = gamma
        accuracies[i] = accuracy
        create_plot(X_train, y_train, rbf)
        plt.show()
    # plt.plot(gammas, accuracies, color='blue')
    # plt.scatter(best_gamma, max_acc, color='red')
    # plt.title('Accuracy as function of gamma')
    # plt.ylabel('Accuracy')
    # plt.xlabel('gamma')
    # plt.xscale('log')
    plt.show()
    return accuracies



if __name__ == '__main__':
    X_train, y_train, X_val, y_val = get_points()
    linear_accuracy_per_C(X_train, y_train,X_val,y_val)
    # print(rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val))
    # print(X_train.var())