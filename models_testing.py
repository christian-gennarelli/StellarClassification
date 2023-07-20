import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def kfold(model, X, y):
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    return cross_val_score(model, X, y, cv=cv, scoring='accuracy')

def hyperparam_test_RFC(X_train, y_train, X_test, y_test, impurity, features):
    scores = []
    range_depths = range(1,8+1) # Range of depths for the trees in the forest
    range_estimators = range(1,8+1) # Range of number of trees in the forest
    for depth in range_depths:
        temp_scores = []
        for num_est in range_estimators:
            RFC = RandomForestClassifier(
                max_depth = depth, # Number of levels in each of the DT
                criterion = impurity, # Impurity function used
                n_estimators = num_est, # Number of trees in the forest
                min_impurity_decrease = 0.001, # Minimum decrease in impurity for being selected as splitting feature/value
                max_features = features, # Number of features to use with each of the decision tree
                random_state = 0
            )

            x = kfold(RFC, X_train, y_train)
            temp_scores.append(x.mean())

            # Model fitting
            #RFC = RFC.fit(X_train, y_train)

            # Model testing
            #y_pred = RFC.predict(X_test)
            #temp_scores.append(metrics.accuracy_score(y_test, y_pred))
        scores.append(temp_scores)
    return scores

def hyperparam_test_DTC(X_train, y_train, X_test, y_test, impurity):
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    depth_range = range(1,8+1)
    #DTC_scores = []
    scores = []
    for depth in depth_range:
        DTC = tree.DecisionTreeClassifier(
            random_state = 0,
            max_depth = depth,
            criterion = impurity,
            min_impurity_decrease = 0.001,
        )
        
        x = kfold(DTC, X_train, y_train)
        scores.append(x.mean())
        
        # Model fitting
        #DTC.fit(X_train, y_train)

        # Model testing
        #y_pred = DTC.predict(X_test)
        #DTC_scores.append(metrics.accuracy_score(y_test, y_pred))
    
    return np.array(scores)