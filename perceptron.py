#################################
# Your name: Noam Koren
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sgd


"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""

def perceptron(data, labels):
    """
	returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    weights = np.zeros(shape=(data.shape[1],))

    # calculate wt * xt. if correct. continue
    for t in range(len(data)):
        product = np.dot(weights, data[t])
        if product >= 0:
            y = 1
        else:
            y = -1
        if y != labels[t]:
            weights = np.add(weights, np.multiply(data[t], -y))

    return weights


#################################

# Place for additional code

#################################

def calc_accuracy(weights, test_data, test_labels):
    labels = np.where(np.dot(test_data, weights) >= 0, 1, -1)
    return np.sum(labels == test_labels) / len(test_data)


def q1a(train_data, train_labels, test_data, test_labels):
    stats = []
    ns = [5, 10, 50, 100, 500, 1000, 5000]
    for n in ns:
        accuracies = []
        for i in range(100):
            idx = np.random.RandomState(i).permutation([x for x in range(n)])
            run_samples = train_data[idx], train_labels[idx]
            w = perceptron(*run_samples)
            accuracy = calc_accuracy(w, test_data, test_labels)
            accuracies.append(accuracy)
        avg = np.average(accuracies)
        perc_5th, perc_95th = np.percentile(accuracies, [5, 95])
        stats.append([avg, perc_5th, perc_95th])

    # fig, ax = plt.subplots()
    # fig.patch.set_visible(False)
    # ax.axis('off')
    # ax.axis('tight')
    # ax.table(cellText=stats, colLabels=['acc', '5th', '95th'], loc='center')
    # fig.tight_layout()
    # plt.savefig('a.png')
    return stats


def q1b(train_data, train_labels):
    w = perceptron(train_data, train_labels)
    # plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    # plt.savefig('c.png')
    return w


def q1c(w, data, labels):
    accuracy = calc_accuracy(w, data, labels)
    print(accuracy)
    return accuracy


def q1d(weights, test_data, test_labels):
    labels = np.where(np.dot(test_data, weights) >= 0, 1, -1)
    errors = np.where(labels != test_labels)
    errors = test_data[errors][:2]
    # for i, err in enumerate(errors):
    #     plt.imshow(np.reshape(err, (28, 28)), interpolation='nearest')
    #     plt.savefig(f'd{i}.png')
    return errors


def main():
    train_data, train_labels, val_data, val_labels, test_data, test_labels = sgd.helper()
    train_data = sklearn.preprocessing.normalize(train_data)
    test_data = sklearn.preprocessing.normalize(test_data)

    q1a(train_data, train_labels, test_data, test_labels)

    w = q1b(train_data, train_labels)
    q1c(w, test_data, test_labels)
    q1d(w, test_data, test_labels)


if __name__ == "__main__":
    main()
