import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA


## METRICS FUNCTIONS
def corr_matrix(df):
    plt.figure(figsize=(15, 10))
    plot = sns.heatmap(df, linewidth=0.25, annot=True)
    plt.show(plot)

def plotLabels(y):
    data = {'Galaxy': sum(1 for el in y if el == 0), 
            'Quasar': sum(1 for el in y if el == 1),
            'Stars': sum(1 for el in y if el == 2)
           }
    classes = list(data.keys())
    values = list(data.values())
  
    fig = plt.figure(figsize = (10, 5))
 
    plt.bar(classes, values, color ='red', width = 0.4)
 
    plt.xlabel("Classes")
    plt.ylabel("Sample size")
    plt.title("Labels distribution in the dataset")
    plt.show()

def plot_confusion_matrix(confusion_matrix):
    confusion_matrix = pd.DataFrame(confusion_matrix)
    plt.figure(figsize=(10, 5))
    s = sns.heatmap(confusion_matrix, annot=True)

def variance_inflation(df):
    vif = pd.DataFrame()
    vif["Variable"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    print(vif)

def entropy(y_train):
    entropy_vals = []
    for num in range(0,3):
        p = sum(1 for el in y_train if el == num)/len(y_train) # p(x)
        entropy_vals.append(p*math.log(p, 2)) # p(x)log2(p(x))
    return entropy_vals

def gini_impurity(y_train):
    gini_vals = []
    for num in range(0,3):
        p = sum(1 for el in y_train if el == num)/len(y_train) # p(x)
        gini_vals.append(p**2) # p(x)^2
    return gini_vals

def plot_scores(scores, model):
    accuracy_matrix = pd.DataFrame(scores)
    plt.figure(figsize=(20, 10))
    s = sns.heatmap(accuracy_matrix, annot=True)
    if model == "DTC":
        s.set(ylabel = 'Depth')
    else:
        s.set(xlabel = 'Number of estimators', ylabel = 'Depth')

def plot_pca_elbow(X):
    n_components = range(1, X.shape[1] + 1)
    explained_variance = []

    for n in n_components:
        pca = PCA(n_components=n)
        pca.fit(X)
        explained_variance.append(pca.explained_variance_ratio_.sum())

    # Plot cumulative explained variance ratio
    plt.plot(n_components, explained_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Elbow Method - Explained Variance')
    plt.show()