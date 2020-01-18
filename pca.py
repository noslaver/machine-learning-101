import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, ShuffleSplit


def plot_vector_as_image(image, h, w, title):
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimensions of original pi
    """
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title(title, size=12)
    #plt.show()
    #plt.savefig(f"{title}.png")

def get_pictures_by_name(name=None):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    selected_images = []
    n_samples, h, w = lfw_people.images.shape

    if name != None:
        target_label = list(lfw_people.target_names).index(name)

    for image, target in zip(lfw_people.images, lfw_people.target):
        if name == None or (target == target_label):
            image_vector = image.reshape((h*w, 1))
            selected_images.append(image_vector)
    return selected_images, h, w

def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
        X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
        For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
        k - number of eigenvectors to return

    Returns:
      U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors 
          of the covariance matrix.
      S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """
    if type(X[0]) is np.ndarray:
        for i in range(len(X)):
            X[i] = [x[0] for x in X[i]]

    X = X - np.mean(X)

    U, S, Vt = np.linalg.svd(X)
    S[:k] = np.power(S[:k], 2)

    return Vt[:k], S[:k]


def q2(images, h, w):
    k = 10

    U, S = PCA(images, k)

    for i, u in enumerate(U):
        plot_vector_as_image(u, h, w, f"Eigen-vector {i}")


def q3(images, h, w):
    ks = [1, 5, 10, 30, 50, 100]
    dists = []
    for k in ks:
        dist = 0
        U, S = PCA(images, k)
        UtU = (U.T).dot(U)

        indices = np.random.randint(0, len(images) + 1, 5)
        for i, img in enumerate(images):
            img = np.array(img)
            trans_img = UtU.dot(img)
            dist += np.linalg.norm(trans_img - img)
            if i in indices:
                plot_vector_as_image(img, h, w, f"Original picture {i} - k - {k}")
                plot_vector_as_image(trans_img, h, w, f"Transformed picture {i} - k - {k}")
        dists.append(dist)


    #plt.clf()
    #plt.plot(ks, dists)
    #plt.title("L2 distances sum as function of k")
    #plt.xlabel("k")
    #plt.ylabel("L2 disatnces sum")
    #plt.savefig("L2-sums.png")


def q4():
    data = load_data()
    images, h, w = get_pictures_by_name()
    y = data.target

    ks = [1, 5, 10, 30, 50, 100, 150, 300]

    param_grid = { "gamma": [1e-7], "C": [1000], "kernel": ["rbf"] }
    accs = []
    for k in ks:
        U, S = PCA(images, k)
        grid = GridSearchCV(SVC(), param_grid, cv=ShuffleSplit(n_splits=2, test_size=0.25, random_state=k), n_jobs=-1)
        UtU = (U.T).dot(U)
        trans_imgs = [UtU.dot(img) for img in images]
        grid.fit(trans_imgs, y)
        accs.append(grid.best_score_)

    # Full picture training
    grid = GridSearchCV(SVC(), param_grid, cv=ShuffleSplit(n_splits=2, test_size=0.25, random_state=k), n_jobs=-1)
    grid.fit(images, y)
    accs.append(grid.best_score_)
    ks.append(1850)

    #plt.clf()
    #plt.plot(ks, accs)
    #plt.title("Classification accuracy as function of  k")
    #plt.xlabel("k")
    #plt.ylabel("Accuracy")
    #plt.savefig("accuracies.png")


if __name__ == "__main__":
    data = load_data()

    images, h, w = get_pictures_by_name("Ariel Sharon")
    q2(images, h, w)
    q3(images, h, w)
    q4()
