from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class FareDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, bins, labels, column='Fare'):
        self.bins = bins
        self.labels = labels
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        X[self.column] = pd.cut(X[self.column], bins=self.bins, labels=self.labels, include_lowest=True)
        return X
    
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols, columns):
        self.drop_cols = drop_cols
        self.columns = columns
    
    def fit(self, X, y=None):
        self.columns_ = pd.get_dummies(X, columns=self.columns).columns
        return self
    
    def transform(self, X):
        X = X.copy()

        X_encoded = pd.get_dummies(X, columns=self.columns, dtype=int)
        X_encoded.drop(labels=[col for col in self.drop_cols if col in X_encoded.columns], axis=1, inplace=True)
        return X_encoded
    
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.scaler = MinMaxScaler()
    
    def fit(self, X, y=None):
        if self.columns is not None:
            self.scaler.fit(X[self.columns])
        else:
            self.scaler.fit(X)
        return self
    
    def transform(self, X):
        X = X.copy()  
        if self.columns is not None:
            X_scaled = self.scaler.transform(X[self.columns])
            X[self.columns] = X_scaled
        else:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)
        return X