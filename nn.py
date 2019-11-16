from collections import Counter
import numpy as np
import numpy.random
import plotly.express as px
# from sklearn.datasets import fetch_openml

# mnist = fetch_openml('mnist_784')
# data = mnist['data']
# labels = mnist['target']

def knn(train_images, labels, test_image, k):
    neighbors = get_neighbors(train_images, test_image, k)
    labels = labels[neighbors]
    return majority(labels)

def get_neighbors(train_images, test_image, k):
    return np.argsort(np.array([dist(image, test_image) for image in train_images]))[:k]

def dist(v1, v2):
    return np.sqrt(np.sum(np.power(v1 - v2, 2)))

def majority(array):
    lst = list(array)
    c = Counter(lst)
    value, count = c.most_common()[0]
    return value

def zo_err(predicted, actual):
    err = np.sum(predicted != actual) / len(predicted)
    return err

# section b
def k_10_error():
    idx = numpy.random.RandomState(0).choice(70000, 1100)
    train = data[idx[:1000], :].astype(int)
    train_labels = labels[idx[:1000]]
    test = data[idx[1000:], :].astype(int)
    test_labels = labels[idx[1000:]]
    prediction = np.array([knn(train, train_labels, t, 10) for t in test])
    print(zo_err(prediction, test_labels))

# section c
def plot_k_1_to_100():
    predictions = []
    errors = []
    idx = numpy.random.RandomState(0).choice(70000, 1100)
    train = data[idx[:1000], :].astype(int)
    train_labels = labels[idx[:1000]]
    test = data[idx[1000:], :].astype(int)
    test_labels = labels[idx[1000:]]
    for k in range(1, 101):
        print(f'calculating error for k = {k}')
        prediction = [knn(train, train_labels, t, k) for t in test]
        error = zo_err(prediction, test_labels)
        print(f'error - {error}')
        errors.append(error)
    fig = px.scatter(x=range(1, 101), y=errors, labels={'x':'k', 'y':'error'})
    fig.show()

# section d
def plot_n_100_to_5000():
    predictions = []
    errors = []
    for n in range(100, 5000, 100):
        print(f'calculating error for n = {n}')
        idx = numpy.random.RandomState(0).choice(70000, int(n + 0.1 * n))
        train = data[idx[:n], :].astype(int)
        train_labels = labels[idx[:n]]
        test = data[idx[n:], :].astype(int)
        test_labels = labels[idx[n:]]
        prediction = [knn(train, train_labels, t, 1) for t in test]
        error = zo_err(prediction, test_labels)
        print(f'error - {error}')
        errors.append(error)
    fig = px.scatter(x=range(100, 5000, 100), y=errors)
    fig.show()

