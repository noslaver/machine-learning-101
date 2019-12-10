#################################
# Your name: Noam Koren
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper():
  mnist = fetch_mldata('MNIST original')
  data = mnist['data']
  labels = mnist['target']

  neg, pos = 0,8
  train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
  test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

  train_data_unscaled = data[train_idx[:6000], :].astype(np.float64)
  train_labels = (labels[train_idx[:6000]] == pos)*2-1

  validation_data_unscaled = data[train_idx[6000:], :].astype(np.float64)
  validation_labels = (labels[train_idx[6000:]] == pos)*2-1

  test_data_unscaled = data[60000+test_idx, :].astype(np.float64)
  test_labels = (labels[60000+test_idx] == pos)*2-1

  # Preprocessing
  train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
  validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
  test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
  return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def SGD(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the final classifier
    """
    weights = np.zeros(shape=(data.shape[1],))

    for t in range(1, T + 1):
        idx = np.random.randint(0, len(data) - 1)
        y = labels[idx]
        x = data[idx]
        if y * np.dot(weights, x) < 1:
            eta = eta_0 / t
            weights = (1 - eta) * weights + eta * C * y * x
        else:
            weights = (1 - eta) *  weights

    return weights


#################################

# Place for additional code

#################################

def q2a(data, labels, val_data, val_labels):
    T = 1000
    C = 1

    stats = []
    xs = list(range(-5, 6))
    for x in xs:
        eta_0 = 10**x
        accuracies = []
        for i in range(10):
            w = SGD(data, labels, C, eta_0, T)
            accuracy = calc_accuracy(w, val_data, val_labels)
            accuracies.append(accuracy)
        avg = np.average(accuracies)
        stats.append(avg)

    # plt.plot(xs, stats)
    # plt.ylabel('Prediction accuracy')
    # plt.xlabel('log(eta_0)')
    # plt.savefig('2aa.png')

    stats = []
    xs = [-1 + i / 10 for i in range(20)]
    for x in xs:
        eta_0 = 10**x
        accuracies = []
        for i in range(10):
            w = SGD(data, labels, C, eta_0, T)
            accuracy = calc_accuracy(w, val_data, val_labels)
            accuracies.append(accuracy)
        avg = np.average(accuracies)
        stats.append(avg)

    # plt.plot(xs, stats)
    # plt.ylabel('Prediction accuracy')
    # plt.xlabel('log(eta_0)')
    # plt.savefig('2ab.png')
    return 10**xs[np.argmax(stats)]


def q2b(data, labels, val_data, val_labels):
    T = 1000
    eta_0 = 0.63096

    stats = []
    xs = list(range(-7, 5))
    for x in xs:
        C = 10**x
        accuracies = []
        for i in range(10):
            w = SGD(data, labels, C, eta_0, T)
            accuracy = calc_accuracy(w, val_data, val_labels)
            accuracies.append(accuracy)
        avg = np.average(accuracies)
        stats.append(avg)

    # plt.plot(xs, stats)
    # plt.ylabel('Prediction accuracy')
    # plt.xlabel('log(eta_0)')
    # plt.savefig('2ba.png')

    stats = []
    xs = [-5 + i / 10 for i in range(20)]
    for x in xs:
        C = 10**x
        accuracies = []
        for i in range(10):
            w = SGD(data, labels, C, eta_0, T)
            accuracy = calc_accuracy(w, val_data, val_labels)
            accuracies.append(accuracy)
        avg = np.average(accuracies)
        stats.append(avg)

    # plt.plot(xs, stats)
    # plt.ylabel('Prediction accuracy')
    # plt.xlabel('log(eta_0)')
    # plt.savefig('2bb.png')
    return 10**xs[np.argmax(stats)]


def q2c(data, labels, eta_0, C):
    T = 20000

    w = SGD(data, labels, C, eta_0, T)
    # plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    # plt.savefig('2c.png')

    return w


def q2d(w, data, labels):
    return calc_accuracy(w, data, labels)


def calc_accuracy(weights, data, labels):
    pred = np.where(np.dot(data, weights) >= 0, 1, -1)
    return np.sum(pred == labels) / len(data)


def main():
    train_data, train_labels, val_data, val_labels, test_data, test_labels = helper()

    best_eta = q2a(train_data, train_labels, val_data, val_labels)
    # print(best_eta)

    best_C = q2b(train_data, train_labels, val_data, val_labels)
    # print(best_C)

    w = q2c(train_data, train_labels, best_eta, best_C)

    acc = q2d(w, test_data, test_labels)
    # print(acc)


if __name__ == "__main__":
    main()
