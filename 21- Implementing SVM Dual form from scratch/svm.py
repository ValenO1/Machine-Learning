import numpy as np


class LinearSVM:
    def __init__(self, learning_rate=0.001, epochs=1000, C=1, dual=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.weights = None
        self.bias = 0
        self.dual = dual

    def primal_form(self, X, y):
        for i in range(self.epochs):
            condition = y * (X.dot(self.weights) + self.bias)
            idx_misclassified_data = np.where(condition < 1)[0]

            gradient_weights = self.weights - self.C * y[idx_misclassified_data].dot(
                X[idx_misclassified_data]
            )
            self.weights -= self.learning_rate * gradient_weights

            gradient_bias = -self.C * np.sum(y[idx_misclassified_data])
            self.bias -= self.learning_rate * gradient_bias

    def dual_form(self, X, y):
        self.get_alpha(X, y)
        indices_sv = [i for i in range(len(X)) if self.alpha[i] != 0]
        for i in indices_sv:
            self.weights += self.alpha[i] * y[i] * X[i]

        for i in indices_sv:
            self.bias += y[i] - np.dot(self.weights.T, X[i])
        self.bias /= len(indices_sv)

    def get_alpha(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        for _ in range(self.epochs):
            y = y.reshape(-1, 1)
            H = (y.dot(y.T)) * (X.dot(X.T))
            gradient = np.ones(n_samples) - H.dot(self.alpha)
            self.alpha += self.learning_rate * gradient
        self.alpha = np.where(self.alpha < 0, 0, self.alpha)
        self.alpha = np.where(self.alpha > self.C, self.C, self.alpha)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        if self.dual:
            self.dual_form(X, y)
        else:
            self.primal_form(X, y)

    def predict(self, X):
        location = X.dot(self.weights) + self.bias
        return np.where(location >= 0, 1, -1)

    def score(self, X, y):
        prediction = self.predict(X)
        return np.mean(y == prediction)
