import numpy as np
from itertools import combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, degree= 2, ):
        self.degree = degree
        
    def fit_transform(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        poly_features = np.ones((n_samples, 1))
        for d in range(1, self.degree +1 ):
            for comb in combinations_with_replacement(range(n_features),d):
                new_feature = np.prod(X[:, comb], axis = 1, keepdims=True)
                poly_features = np.c_[poly_features, new_feature]
        
        return poly_features
    
class MinMaxScaler():
    def _init__(self):
        self.min_val = None
        self.max_val = None
        
    def fit(self, X):
        self.min_val = X.min(axis = 0)
        self.max_val = X.max(axis = 0)
        
    def transform(self,X):
        if self.min_val is None or self.max_val is None:
            raise ValueError("please fit before applying transform")
        X_normalized = (X - self.min_val) / (self.max_val - self.min_val)   
        return X_normalized        
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class StandardScaler():
    def _init__(self):
        self.mean_value = None
        self.std_value = None
        
    def fit(self, X):
        self.mean_value = np.mean(X, axis= 0)
        self.std_value = np.std(X, axis=0)
        
    def transform(self,X):
        if self.mean_value is None or self.std_value is None:
            raise ValueError("please fit before applying transform")
        X_standardized = (X - self.mean_value) / self.std_value  
        return X_standardized        
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)