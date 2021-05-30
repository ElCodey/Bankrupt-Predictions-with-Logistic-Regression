import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.decomposition import PCA

def preprocessing(df):
    df = df.copy()
    
    X = df.drop("Bankrupt?", axis=1)
    y = df["Bankrupt?"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    
    return X_train, X_test, y_train, y_test

def log_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_pred

def oversampler(X_train, y_train):
    oversampler = RandomOverSampler(random_state=1)
    X_train_os, y_train_os = oversampler.fit_resample(X_train, y_train)
    
    return X_train_os, y_train_os

def smote(X_train, y_train):
    oversampler = SMOTE(random_state=1)
    X_train_smote, y_train_smote = oversampler.fit_resample(X_train, y_train)
    
    return X_train_smote, y_train_smote

def pca(X_train, X_test):
    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    X_train_pca = pd.DataFrame(pca.transform(X_train))
    X_test_pca = pd.DataFrame(pca.transform(X_test))
    
    return X_train_pca, X_test_pca