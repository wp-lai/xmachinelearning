import numpy as np
from logistic import LogisticRegression


def read_data():
    # data from:
    # http://cs229.stanford.edu/ps/ps1/logistic_x.txt
    # http://cs229.stanford.edu/ps/ps1/logistic_y.txt
    try:
        X = np.loadtxt('logistic_x.txt')
    except FileNotFoundError:
        X = np.loadtxt('datas/binary_classification/logistic_x.txt')

    try:
        y = np.loadtxt('logistic_y.txt')
    except FileNotFoundError:
        y = np.loadtxt('datas/binary_classification/logistic_y.txt')

    return (X, y)


def test_lr_newton_method():
    X, y = read_data()

    lr_clf = LogisticRegression(solver="newton_method")
    lr_clf.fit(X, y)

    # test intercept
    intercept = lr_clf.intercept_
    assert(abs(intercept - -2.618) < 0.01)

    # test coefficient
    coef = lr_clf.coef_
    assert(abs(coef[0] - 0.76) < 0.01)
    assert(abs(coef[1] - 1.17) < 0.01)


def test_lr_gradient_descent():
    X, y = read_data()

    lr_clf = LogisticRegression(learning_rate=0.1, max_iter=10000,
                                solver="gradient_descent")
    lr_clf.fit(X, y)

    # test intercept
    intercept = lr_clf.intercept_
    assert(abs(intercept - -2.618) < 0.01)

    # test coefficient
    coef = lr_clf.coef_
    assert(abs(coef[0] - 0.76) < 0.01)
    assert(abs(coef[1] - 1.17) < 0.01)


def test_lr_stochastic_gradient_descent():
    X, y = read_data()

    lr_clf = LogisticRegression(learning_rate=0.001, max_iter=10000,
                                solver="stochastic_gradient_descent")
    lr_clf.fit(X, y)

    # test intercept
    intercept = lr_clf.intercept_
    assert(abs(intercept - -2.618) < 0.01)

    # test coefficient
    coef = lr_clf.coef_
    assert(abs(coef[0] - 0.76) < 0.01)
    assert(abs(coef[1] - 1.17) < 0.01)


if __name__ == "__main__":
    test_lr_newton_method()
    test_lr_gradient_descent()
    test_lr_stochastic_gradient_descent()
