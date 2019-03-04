import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D

def load_data():
    # load data 
    data = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', header=None)
    p_ = data.shape[1]
    p = p_ - 1
    X = data.iloc[:, :p]
    y = data.iloc[:, p]

    # split the data to training set and testing test
    # the training set will be used to choose hyperparameters
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    return X_train.values, X_test.values, y_train.values, y_test.values


class XGBOOST():
    def __init__(self, min_impurity_decrease=0.01, M=5):
        self.min_impurity_decrease = min_impurity_decrease
        self.M = M
        
        # initialize regression trees
        self.trees = []
        for _ in range(M):
            tree = DecisionTreeRegressor(min_impurity_decrease \
                                         = self.min_impurity_decrease)
            self.trees.append(tree)

    def fit(self, train_features, train_labels):
        n = train_features.shape[0]
    
        y_pred = np.zeros((n))
        for iter in range(self.M):
            tree = self.trees[iter]
            
            yhat = np.clip(y_pred, 1e-15, 1-1e-15)
            p = 1.0/(1.0 + np.exp(-yhat))
            # loss = -y * np.log(p) - (1-y) * np.log(1-p)
            G = p - train_labels # 1st derivative
            H = p * (1-p) # 2nd derivative
                
            train_labels_new = -G/H
            tree.fit(train_features, train_labels_new, sample_weight = H)
            update_pred = tree.predict(train_features)
            y_pred += update_pred


    def scoring(self, test_features, test_labels):
        y_pred = None
        
        # make predictions
        for tree in self.trees:
            # estimate gradient and update prediction
            update_pred = tree.predict(test_features)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred += update_pred
            
            y_pred = 1.0/(1.0 + np.exp(-y_pred))
            y_pred = (y_pred > 0.5)
            return y_pred

            pass

#########################################################


# define a function to do cross-validation to search for hyperparameter values with training set
def cross_validation(log_min_impurity_decreases, Ms, n_splits=5):
    
    X_train, X_test, y_train, y_test = load_data()

    len_impurity = len(log_min_impurity_decreases)
    len_M = len(Ms)
    kf = KFold(n_splits = 5)
    fold_size = int(len(X_train)/n_splits)
    plot_values_list = []
    
    for m in range(len_impurity):
        for k in range(len_M):
            acc = 0
            ypred = XGBOOST(min_impurity_decrease = \
                        np.exp(log_min_impurity_decreases[m]),M = Ms[k])
            for train_index, test_index in kf.split(X_train):
                traindata, trainlabel = X_train[train_index], y_train[train_index]
                testdata, testlabel = X_train[test_index], y_train[test_index]
                ypred.fit(train_features = traindata, train_labels = trainlabel)
                yvalid = ypred.scoring(test_features = testdata,\
                                                   test_labels = testlabel)
                acc += np.sum(yvalid == testlabel)/fold_size
            mean_acc = acc/n_splits
#         print("mean accuracy when log_id is %d and M is %d: "\
#                          %(log_min_impurity_decreases[m], Ms[k]), mean_acc)
            plot_values_list.append([log_min_impurity_decreases[m], Ms[k], mean_acc])

    plot_values = np.array(plot_values_list)
    index = np.argmax(plot_values[:,2])
    best_log_min_impurity_decrease = plot_values[index,0]
    best_M = int(plot_values[index,1])

    #########################################################


    return plot_values, best_log_min_impurity_decrease, best_M

def run():

    X_train, X_test, y_train, y_test = load_data()
    log_min_impurity_decreases = list(range(-8, 0))
    Ms = list(range(20, 50, 2))
    plot_values, best_log_min_impurity_decrease, best_M = cross_validation(log_min_impurity_decreases=log_min_impurity_decreases, Ms=Ms)

    # plot the accuracy over the hyperparameter values
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(plot_values[:, 0], plot_values[:, 1], plot_values[:, 2], linewidth=0.2, antialiased=True)
    plt.show()


    # fit the train data with the chosen hyperparameters
 
    xgb = XGBOOST(min_impurity_decrease = np.exp(best_log_min_impurity_decrease),\
              M = best_M)
    xgb.fit(train_features = X_train, train_labels = y_train)

    # print the testing accuracy with the chosen models

    ypred = xgb.scoring(test_features = X_test,test_labels = y_test)
    acc = np.sum(ypred == y_test)/len(y_test)
    print("The chosen parameters are: ")
    print("Best log_min_impurity_decrease: ", best_log_min_impurity_decrease)
    print("Best number of trees M: ", best_M)
    print("The testing accuracy with the chosen model: ", acc)





