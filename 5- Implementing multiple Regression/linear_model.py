import numpy as np

class LinearRegression:
    def __init__(self):
        self.betas = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        bias_term = np.ones((n_samples, 1))
        X = np.c_[bias_term, X]
        self.betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.y_mean = np.mean(y)
        
    def predict(self, X_new):
        n_samples = X_new.shape[0]
        bias_term = np.ones((n_samples, 1))
        X_new = np.c_[bias_term, X_new]
        prediction = X_new.dot(self.betas)
        return prediction
    
    def score(self, X, y):
        y_pred = self.predict(X)
        SSE = np.sum((y- y_pred) **2)
        SST = np.sum((y - self.y_mean)**2)
        r_squared = 1 - (SSE / SST)
        return r_squared