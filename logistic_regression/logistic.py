"""
Logistic Regression
"""
import numpy as np
from numpy.linalg import inv


class LogisticRegression:
    """Logistic Regression classifier

    Parameters
    ----------
    learning_rate : float, default 0.01
        learning rate for gradient descent.

    max_iter : int, default 10000
        Maximum number of iterations.

    solver : {'gradient_descent', 'newton_method',
              'stochastic_gradient_descent'}, default: 'newton_method'
        Algorithm to use in the optimization problem.

    tol : float, default: 1e-4
        Tolerance for stopping creteria.
        stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    Attributes
    ----------
    coef_ : array, shape (n_features, )
        Coefficient of the features in the decision function.

    intercept_ : float
        Intercept added to the decision function.
    """

    def __init__(self,
                 learning_rate=0.01,
                 max_iter=10000,
                 solver="newton_method",
                 tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.solver = solver
        self.tol = tol

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array, shape (n_samples,)
            Target vector relative to X. Positive target encoded to 1 and
            negative target encoded to -1.

        Returns
        -------
        self : object
            return self.
        """
        n_samples, n_features = X.shape
        X = np.hstack((np.ones((n_samples, 1)), X))  # add constant
        self.theta = np.zeros(n_features + 1)  # init theta

        if self.solver == "gradient_descent":
            for itr in range(self.max_iter):
                margins = y * np.dot(X, self.theta)
                probs = 1 / (1 + np.exp(-margins))
                self.loss = -(1 / n_samples) * np.sum(np.log(probs))

                grad = -(1 / n_samples) * np.dot(X.T, (1 - probs) * y)
                if all(grad < self.tol):
                    break

                # theta = theta - alpha * gradient
                self.theta -= self.learning_rate * grad

        elif self.solver == "stochastic_gradient_descent":
            for itr in range(self.max_iter):
                order = np.random.permutation(range(n_samples))
                for index in order:
                    data = X[index, :]
                    label = y[index]
                    margins = label * data.dot(self.theta)
                    probs = 1 / (1 + np.exp(-margins))
                    self.theta -= \
                        - self.learning_rate * (1 - probs) * label * data

        elif self.solver == "newton_method":
            for itr in range(self.max_iter):
                margins = y * np.dot(X, self.theta)
                probs = 1 / (1 + np.exp(-margins))
                self.loss = -(1 / n_samples) * np.sum(np.log(probs))

                grad = -(1 / n_samples) * np.dot(X.T, (1 - probs) * y)
                if all(grad < self.tol):
                    break

                H = (1 / n_samples) \
                    * X.T.dot(np.diag(probs * (1 - probs))).dot(X)
                # theta = theta - inv(Hessian) * gradient
                self.theta -= inv(H).dot(grad)

        return self

    @property
    def intercept_(self):
        return self.theta[0]

    @property
    def coef_(self):
        return self.theta[1:]

    def predict_proba(self, X):
        """Probability estimates

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        T : array, shape (n_samples, 2)
            Return of probability of the sample for positive target and
            negative target in the model.
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # add constant
        margins = 1 * X.dot(self.theta)
        probs_pos = 1 / (1 + np.exp(-margins))
        probs_neg = 1 - probs_pos
        return np.hstack((probs_pos[:, np.newaxis], probs_neg[:, np.newaxis]))

    def predict(self, X):
        """Predicted label

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        T : array, shape (n_samples, )
            Return the label prediction for the sample.
        """
        probs = self.predict_proba(X)
        return np.where(probs[:, 0] > probs[:, 1], 1, -1)
