#################################
# Your name: Noam Koren
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

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


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    n_support = np.ndarray((3,2))
    C = 1000
    svms = [{ 'kernel': 'linear' }, { 'kernel': 'poly', 'degree': 2 }, { 'kernel': 'rbf' }]
    for i, s in enumerate(svms):
        s['C'] = C
        s['gamma'] = 'auto'
        clf = svm.SVC(**s)
        clf.fit(X_train, y_train)
        n_support[i] = clf.n_support_
        # create_plot(X_val, y_val, clf)
        # plt.savefig(f'1a-{s["kernel"]}.png')

    return n_support

def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    scores_val = np.ndarray((11,))
    scores_train = np.ndarray((11,))
    cs = list(range(-5, 6))
    for i, c in enumerate(cs):
        C = 10 ** c
        clf = svm.SVC(kernel='linear', gamma='auto', C=C)
        clf.fit(X_train, y_train)
        pred_val = clf.predict(X_val)
        score_val = accuracy_score(pred_val, y_val)
        scores_val[i] = score_val
        pred_train = clf.predict(X_train)
        score_train = accuracy_score(pred_train, y_train)
        scores_train[i] = score_train

        # if c == 2 or c == -3:
        #     create_plot(X_val, y_val, clf)
        #     plt.savefig(f'1b-{c}.png')

    # plt.plot(cs, scores_val, label='Validation')
    # plt.plot(cs, scores_train, label='Training')
    # plt.legend()
    # plt.ylabel('Prediction accuracy')
    # plt.xlabel('log(C)')
    # plt.savefig('1b-scores.png')

    return scores_val


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C = 10
    scores_val = np.ndarray((11,))
    scores_train = np.ndarray((11,))
    gammas = list(range(-5, 6))
    for i, g in enumerate(gammas):
        gamma = 10 ** g
        clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        clf.fit(X_train, y_train)
        pred_val = clf.predict(X_val)
        score_val = accuracy_score(pred_val, y_val)
        scores_val[i] = score_val
        pred_train = clf.predict(X_train)
        score_train = accuracy_score(pred_train, y_train)
        scores_train[i] = score_train

        if g == 1 or g == 2 or g == -3:
            create_plot(X_val, y_val, clf)
            plt.savefig(f'1c-{g}.png')

    # plt.plot(gammas, scores_val, label='Validation')
    # plt.plot(gammas, scores_train, label='Training')
    # plt.legend()
    # plt.ylabel('Prediction accuracy')
    # plt.xlabel('log(gamma)')
    # plt.savefig('1c-scores.png')

    return_val = scores_val

    scores_val = np.ndarray((20,))
    scores_train = np.ndarray((20,))
    gammas = [-0.75 + i / 10 for i in range(20)]
    for i, g in enumerate(gammas):
        gamma = 10 ** g
        clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        clf.fit(X_train, y_train)
        pred_val = clf.predict(X_val)
        score_val = accuracy_score(pred_val, y_val)
        scores_val[i] = score_val
        pred_train = clf.predict(X_train)
        score_train = accuracy_score(pred_train, y_train)
        scores_train[i] = score_train

    print(scores_val)
    print(gammas)
    # plt.clf()
    # plt.plot(gammas, scores_val, label='Validation')
    # plt.plot(gammas, scores_train, label='Training')
    # plt.legend()
    # plt.ylabel('Prediction accuracy')
    # plt.xlabel('log(gamma)')
    # plt.savefig('1c-scores-2.png')

    return return_val

if __name__ == '__main__':
    points = get_points()

    # support = train_three_kernels(*points)
    # print(support)

    # scores = linear_accuracy_per_C(*points)
    # print(scores)

    scores = rbf_accuracy_per_gamma(*points)
    print(scores)
