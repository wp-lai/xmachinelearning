import numpy as np
from logistic import LogisticRegression


def test_lr_classifier():
    # data from:
    # http://cs229.stanford.edu/ps/ps1/logistic_x.txt
    # http://cs229.stanford.edu/ps/ps1/logistic_y.txt
    try:
        X = np.loadtxt('logistic_x.txt')
    except FileNotFoundError:
        X = np.loadtxt('logistic_regression/logistic_x.txt')

    try:
        y = np.loadtxt('logistic_y.txt')
    except FileNotFoundError:
        y = np.loadtxt('logistic_regression/logistic_y.txt')

    lr_clf = LogisticRegression()
    lr_clf.fit(X, y)

    # test intercept
    intercept = lr_clf.intercept_
    assert(abs(intercept - -2.618) < 0.01)

    # test coefficient
    coef = lr_clf.coef_
    assert(abs(coef[0] - 0.76) < 0.01)
    assert(abs(coef[1] - 1.17) < 0.01)


if __name__ == "__main__":
    test_lr_classifier()
