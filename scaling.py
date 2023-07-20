import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import StandardScaler

def scale_dataset(df, balance=True):
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if balance:
        smotetomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        X, y = smotetomek.fit_resample(X, y)
        
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y