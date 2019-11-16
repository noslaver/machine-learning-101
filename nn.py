from collections import Counter
import numpy as np

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
