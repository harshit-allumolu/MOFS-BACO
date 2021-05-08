"""
    Author : Allumolu Harshit
    Email ID : aharshit3@student.nitw.ac.in

    Evaluation of constructed subsets of feature set
    using k-nn classification in scikit-learn
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import numpy as np


def features(x, solution):
    """
        Function name : features
        Arguments : 
            -- x : Input features
            -- solution : A binary array to select features
        Returns : 
            -- x_temp : The selected input features
    """

    temp = []
    for i in range(len(solution)):
        if solution[i] == 1:
            temp.append(x[:,i:i+1])
    if len(temp) == 0:
        return temp
    x_temp = np.concatenate(temp,axis=1)
    return x_temp


def evaluation(x, y, lambda_,k):
    """
        Function name : evaluation
        Arguments :
            -- x : input features
            -- y : input types/classes
            -- lambda_ : A variable to control the effect of #features in fitness function
            -- k : number of neighbours in knn
        Purpose : Evaluation of constructed subsets using
                LOOCV in k-nn and mean squared error
        Returns : 
            -- acc : Accuracy on the given dataset
    """

    if len(x) == 0:
        return 0,0
    
    # leave one out cross validator
    cv = LeaveOneOut()
    # to maintain the output
    y_true, y_pred = list(), list()

    # loop start
    for train_ix, test_ix in cv.split(x):
        # train test split using cv
        x_train, x_test = x[train_ix,:], x[test_ix,:]
        y_train, y_test = y[train_ix], y[test_ix]
        
        # fit Knn classifier
        model = KNeighborsClassifier(n_neighbors=k)     # tuning required
        model.fit(x_train, y_train)

        # predict and store the output
        y_hat = model.predict(x_test)
        y_true.append(y_test[0])
        y_pred.append(y_hat[0])
    
    # accuracy calculation
    acc = accuracy_score(y_true,y_pred)

    # fitness value
    fitness = (acc*10)**2 / (1 + lambda_ * len(x[0]))

    return fitness, acc
