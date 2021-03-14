"""
    Author : Allumolu Harshit
    Email ID : aharshit3@student.nitw.ac.in

    Evaluation of constructed subsets of feature set
    using k-nn classification in scikit-learn
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error


def evaluation(x, y):
    """
        Function name : evaluation
        Arguments :
            -- x : input features
            -- y : input types/classes
        Purpose : Evaluation of constructed subsets using
                LOOCV in k-nn and mean squared error
    """

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
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(x_train, y_train)

        # predict and store the output
        y_hat = model.predict(x_test)
        y_true.append(y_test[0])
        y_pred.append(y_hat[0])
    
    # error calculation
    error = mean_squared_error(y_true, y_pred)
    return error