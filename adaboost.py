#################################
# Your name: Noam Koren
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    n = len(X_train)
    hypotheses = []
    alphas = []
    Ds = np.array([1 / n] * n)
    for t in range(1, T + 1):
        h_t, err = weak_learner(Ds, X_train, y_train)
        alpha_t = 0.5 * np.log((1 - err) / err)
        preds = [predict(h_t, x) for x in X_train]
        exps = np.exp(-alpha_t * np.multiply(preds, y_train))
        mul = np.multiply(Ds, exps)
        Ds = mul / np.sum(mul)
        alphas.append(alpha_t)
        hypotheses.append(h_t)

    return hypotheses, alphas


def weak_learner(weights, X_train, y_train):
    neg_h, neg_err = _weak_learner(weights, X_train, y_train, 1)
    pos_h, pos_err = _weak_learner(weights, X_train, y_train, -1)

    if neg_err < pos_err:
        return neg_h, neg_err
    else:
        return pos_h, pos_err


def _weak_learner(weights, X_train, y_train, b):
    import sys
    h = -1, 0 # index, theta
    best_err = sys.maxsize
    data = list(zip(X_train, y_train, weights))

    n = len(X_train)
    dim = len(X_train[0])

    for i in range(dim):
        err = np.sum([w for _, y, w in data if y == b])
        data.sort(key=lambda d: d[0][i])

        if err < best_err:
            best_err = err
            h = i, data[0][0][i] - 1

        for j in range(n):
            x, y, w = data[j]
            err -= b * y * w

            if err < best_err:
                if j == n:
                    best_err = err
                    h = i, 0.5 + x[i]
                elif j != n - 1 and x[i] != data[j + 1][0][i]:
                    best_err = err
                    h = i, 0.5 * (x[i] + data[j + 1][0][i])

    best_i, best_theta = h
    return (b, best_i, best_theta), best_err


def predict(h, x):
    h_pred, h_index, h_theta = h
    if x[h_index] <= h_theta:
        return h_pred 
    else:
        return -h_pred


def error(hypotheses, alphas, X, y):
    err = 0
    for x, y in zip(X, y):
        pred = np.sign(np.sum([predict(h, x) * w for h, w in zip(hypotheses, alphas)]))
        if pred != y:
            err += 1

    return err / len(X)


def exp_loss(hypotheses, alphas, X, y):
    loss = 0
    for x, y in zip(X, y):
        loss += np.exp(-y * np.sum([predict(h, x) * w for h, w in zip(hypotheses, alphas)]))

    return loss / len(X)


##############################################
# You can add more methods here, if needed.

def qa(hypotheses, alpha_vals, X_train, y_train, X_test, y_test, T):
    Ts = list(range(1, T + 1))
    train_errs = []
    test_errs = []
    for t in Ts:
        t_hypotheses = hypotheses[:t]
        t_alphas = alpha_vals[:t]
        train_err = error(t_hypotheses, t_alphas, X_train, y_train)
        train_errs.append(train_err)
        test_err = error(t_hypotheses, t_alphas, X_test, y_test)
        test_errs.append(test_err)

    # plt.plot(Ts, train_errs)
    # plt.xlabel('t')
    # plt.ylabel('Error')
    # plt.title('Training error')
    # plt.savefig('a-train.png')

    # plt.clf()
    # plt.plot(Ts, test_errs)
    # plt.xlabel('t')
    # plt.ylabel('Error')
    # plt.title('Test error')
    # plt.savefig('a-test.png')


def qb(X_train, y_train, vocab):
    T = 10

    hypotheses, _ = run_adaboost(X_train, y_train, T)
    # print(list((h, vocab[h[1]]) for h in hypotheses))


def qc(hypotheses, alpha_vals, X_train, y_train, X_test, y_test, T):
    Ts = list(range(1, T + 1))
    train_losses = []
    test_losses = []
    for t in Ts:
        t_hypotheses = hypotheses[:t]
        t_alphas = alpha_vals[:t]
        train_loss = exp_loss(t_hypotheses, t_alphas, X_train, y_train)
        train_losses.append(train_loss)
        test_loss = exp_loss(t_hypotheses, t_alphas, X_test, y_test)
        test_losses.append(test_loss)

    # plt.clf()
    # plt.plot(Ts, train_losses)
    # plt.xlabel('t')
    # plt.ylabel('Exponential loss')
    # plt.title('Training exponential loss')
    # plt.savefig('c-train.png')

    # plt.clf()
    # plt.plot(Ts, test_losses)
    # plt.xlabel('t')
    # plt.ylabel('Exponential loss')
    # plt.title('Test exponential loss')
    # plt.savefig('c-test.png')

##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    T = 80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    qa(hypotheses, alpha_vals, X_train, y_train, X_test, y_test, T)
    qb(X_train, y_train, vocab)
    qc(hypotheses, alpha_vals, X_train, y_train, X_test, y_test, T)


if __name__ == '__main__':
    main()

