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