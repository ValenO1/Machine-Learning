import numpy as np

class LinearSVM():
    def __init__(self, learning_rate = .001, epochs = 1000, C = 1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.weights = None
        self.bias = 0
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for i in range(self.epochs):
            condition = y * (X.dot(self.weights) + self.bias)
            idx_misclassified_data = np.where(condition < 1)[0]
            
            gradient_weights = self.weights - self.C * y[idx_misclassified_data].dot(X[idx_misclassified_data])
            self.weights -= self.learning_rate * gradient_weights
            
            gradient_bias = -self.C * np.sum(y[idx_misclassified_data])
            self.bias -= self.learning_rate * gradient_bias
            
    def predict(self, X):
        location = X.dot(self.weights) + self.bias
        return np.where(location >= 0, 1, -1)
    
    def score(self, X, y):
        prediction = self.predict(X)
        return np.mean(y == prediction)
        