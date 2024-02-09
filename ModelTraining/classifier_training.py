# -*- coding: utf-8 -*-
"""
PyCharm
@project_name: MOF_CO2_Cycloaddition_Screening
@File        : classifier_training.py
@Author      : Xuefeng-Bai@BJUT
@Date        : 2023/2/8 14:07
"""

"""
This is a grid search for the super parameters of the training program, 
in the model of all the algorithms are encapsulated in different functions, 
by calling the function to the grid, the results will be saved to the.txt document.
"""

# 导入所需包
# Algorithm import
from sklearn import tree
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Other package import
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import precision_score, confusion_matrix, recall_score, f1_score
import numpy as np
import os

# Decision Tree
# 决策树
def dt_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    param_grid = [
        {
            'criterion': ['entropy'],
            'splitter': ["best"],
            'max_depth': [max_depth for max_depth in range(2, 41, 2)],
            'max_leaf_nodes': [max_leaf_nodes for max_leaf_nodes in range(3, 31, 3)]
        },
    ]
    dt_clf = tree.DecisionTreeClassifier()
    grid_search_class = GridSearchCV(dt_clf, param_grid, n_jobs=-1, verbose=4, cv=3)
    grid_search_class.fit(X_train, Y_train)
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/dt_cv_data.csv',
                                                                       mode="w")
    dt_clf = grid_search_class.best_estimator_
    Y_predict = dt_clf.predict(X_test)
    sc = dt_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/dt.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# k-Nearest Neighbors
# k-近邻
def knn_fit(X_train, X_test, Y_train, Y_test):
    try:
        # Hyperparameter search
        param_grid = [
            {
                'weights': ['uniform'],
                'n_neighbors': [n_neighbors for n_neighbors in range(1, 20)]
            },
            {
                'weights': ['distance'],
                'n_neighbors': [n_neighbors for n_neighbors in range(1, 20)],
                'p': [p for p in range(1, 6)]
            }
        ]
        knn_clf = neighbors.KNeighborsClassifier()
        grid_search_class = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=4, cv=3, )
        grid_search_class.fit(X_train, Y_train)
        a = [x for x in grid_search_class.cv_results_.values()]
        pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/knn_cv_data.csv',
                                                                           mode="w")
        knn_clf = grid_search_class.best_estimator_
        Y_predict = knn_clf.predict(X_test)
        sc = knn_clf.score(X_test, Y_test)
        cm = confusion_matrix(Y_test, Y_predict)
        ps = precision_score(Y_test, Y_predict)
        rs = recall_score(Y_test, Y_predict)
        f1 = f1_score(Y_test, Y_predict)
        with open(f"./DataResource/Training_Result/knn.txt", mode="w") as f:
            f.write("best_params:" + "\n")
            f.write(str(grid_search_class.best_params_) + "\n")
            f.write("model_score:" + "\n")
            f.write(str(grid_search_class.best_score_) + "\n")
            f.write("best_score:" + "\n" + str(sc) + "\n")
            f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
            f.write("precision_score:" + "\n" + str(ps) + "\n")
            f.write("recall_score:" + "\n" + str(rs) + "\n")
            f.write("f1_score:" + "\n" + str(f1) + "\n")
            f.write("========================================" + "\n")
    except:
        pass

# Logistic Regression
# 逻辑回归分类
def lr_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    param_grid = [
        {
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'penalty': ['l2', 'l1', 'elasticnet', 'none']
        },
    ]
    LR_clf = LogisticRegression()
    grid_search_class = GridSearchCV(LR_clf, param_grid, n_jobs=-1, verbose=4, cv=3)
    grid_search_class.fit(X_train, Y_train)
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/lr_cv_data.csv',
                                                                       mode="w")
    lr_clf = grid_search_class.best_estimator_
    Y_predict = lr_clf.predict(X_test)
    sc = lr_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/lr.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Neural Networks
# 神经网络
def nn_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    hidden_layer_sizes = []
    for i in range(2, 20, 3):
        for j in range(2, 20, 3):
            hidden_layer_sizes.append((i, j))

    param_grid = [
        {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
        },
    ]
    nn_clf = MLPClassifier()
    grid_search_class = GridSearchCV(nn_clf, param_grid, n_jobs=-1, verbose=4, cv=3)
    grid_search_class.fit(X_train, Y_train)
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/nn_cv_data.csv',
                                                                       mode="w")
    nn_clf = grid_search_class.best_estimator_
    Y_predict = nn_clf.predict(X_test)
    sc = nn_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/nn.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Quadratic Discriminant Analysis
# 二次判别分析
def qda_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    param_grid = [
        {
            'store_covariance': [True, False],
        },
    ]
    qda_clf = QuadraticDiscriminantAnalysis()
    grid_search_class = GridSearchCV(qda_clf, param_grid, n_jobs=-1, verbose=0, cv=3)
    grid_search_class.fit(X_train, Y_train)
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/qda_cv_data.csv',
                                                                       mode="w")
    qda_clf = grid_search_class.best_estimator_
    Y_predict = qda_clf.predict(X_test)
    sc = qda_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/qda.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Random Forest
# 随机森林
def rf_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    rfc = RandomForestClassifier(class_weight='balanced')
    rf_param_grid1 = {'n_estimators': [x for x in range(1, 60, 10)],
                      'min_samples_split': [2],
                      'max_features': range(50, 73, 5),
                      'max_depth': range(20, 30, 3),
                      }
    grid_search_class = GridSearchCV(rfc, param_grid=rf_param_grid1, cv=3, scoring="roc_auc", n_jobs=-1, verbose=2)
    grid_search_class.fit(X_train, Y_train)
    rf_clf = grid_search_class.best_estimator_
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/rf_cv_data.csv',
                                                                       mode="w")
    Y_predict = rf_clf.predict(X_test)
    sc = rf_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/rf.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Stochastic Gradient Descent
# 随机梯度下降
def sgd_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    param_grid = [
        {
            'loss': ['hinge', 'log', ],
            'penalty': ['l2', 'l1', 'elasticnet']
        },
    ]
    sgd_clf = SGDClassifier()
    grid_search_class = GridSearchCV(sgd_clf, param_grid, n_jobs=-1, verbose=4, cv=3)
    grid_search_class.fit(X_train, Y_train)
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/sgd_cv_data.csv',
                                                                       mode="w")
    sgd_clf = grid_search_class.best_estimator_
    Y_predict = sgd_clf.predict(X_test)
    sc = sgd_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/sgd.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Support Vector Machines
# 支持向量机
def svm_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    param_grid = [
        {
            'kernel': ['linear'],
        },
        {
            'kernel': ['rbf', 'sigmoid'],
        },
        {
            'kernel': ['poly'],
        }
    ]
    svm_clf = SVC()
    grid_search_class = GridSearchCV(svm_clf, param_grid, n_jobs=-1, verbose=4, cv=3)
    grid_search_class.fit(X_train, Y_train)
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/svm_cv_data.csv',
                                                                       mode="w")
    svm_clf = grid_search_class.best_estimator_
    Y_predict = svm_clf.predict(X_test)
    sc = svm_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)

    with open(f"./DataResource/Training_Result/svm.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Naive Bayes(Bernoulli)
# 伯努利型贝叶斯
def bnb_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    param_grid = [
        {
            'fit_prior': [True, False],
            'alpha': [1.0, 0.5, 0.000000001]
        }
    ]
    BernoulliNB_clf = BernoulliNB()
    grid_search_class = GridSearchCV(BernoulliNB_clf, param_grid, n_jobs=-1, verbose=4, cv=3)
    grid_search_class.fit(X_train, Y_train)
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/bnb_cv_data.csv',
                                                                       mode="w")
    nb_clf = grid_search_class.best_estimator_
    Y_predict = nb_clf.predict(X_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    sc = nb_clf.score(X_test, Y_test)
    with open(f"./DataResource/Training_Result/bnb.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Naive Bayes(Multinomial)
# 多项式型贝叶斯
def mnb_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    param_grid = [
        {
            'fit_prior': [True, False],
            'alpha': [1.0, 0.5, 0.000000001]
        }
    ]
    MultinomialNB_clf = MultinomialNB()
    grid_search_class = GridSearchCV(MultinomialNB_clf, param_grid, n_jobs=-1, verbose=4, cv=3)
    grid_search_class.fit(X_train, Y_train)
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/mnb_cv_data.csv',
                                                                       mode="w")
    nb_clf = grid_search_class.best_estimator_
    Y_predict = nb_clf.predict(X_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    sc = nb_clf.score(X_test, Y_test)
    with open(f"./DataResource/Training_Result/mnb.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Adaptive Boosting
# 自适应提升
def ada_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    ada = AdaBoostClassifier()
    ada_param_grid1 = {'n_estimators': [x for x in range(25, 501, 25)],
                       'learning_rate': list(np.arange(0.01, 2, 0.1)),
                       }
    grid_search_class = GridSearchCV(ada, param_grid=ada_param_grid1, cv=3, scoring="roc_auc", n_jobs=-1, verbose=2)
    grid_search_class.fit(X_train, Y_train)
    ada_clf = grid_search_class.best_estimator_
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/ada_cv_data.csv',
                                                                       mode="w")
    Y_predict = ada_clf.predict(X_test)
    sc = ada_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/ada.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Gradient Boosting Decision Tree
# 梯度提升决策树
def gbdt_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    gbdt = GradientBoostingClassifier()
    gbdt_param_grid1 = {'n_estimators': [x for x in range(25, 501, 25)],
                        'learning_rate': list(np.arange(0.01, 2, 0.1)),
                        }
    grid_search_class = GridSearchCV(gbdt, param_grid=gbdt_param_grid1, cv=3, scoring="roc_auc", n_jobs=-1, verbose=2)
    grid_search_class.fit(X_train, Y_train)
    gbdt_clf = grid_search_class.best_estimator_
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/gbdt_cv_data.csv',
                                                                       mode="w")
    Y_predict = gbdt_clf.predict(X_test)
    sc = gbdt_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/gbdt.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Extra Trees
# 额外树
def et_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    et = ExtraTreesClassifier()
    et_param_grid1 = {'n_estimators': [x for x in range(25, 301, 25)],
                      'min_samples_split': [2],
                      'max_features': range(2, 73, 10),
                      'max_depth': range(3, 22, 3),
                      }
    grid_search_class = GridSearchCV(et, param_grid=et_param_grid1, cv=3, scoring="roc_auc", n_jobs=-1, verbose=2)
    grid_search_class.fit(X_train, Y_train)
    et_clf = grid_search_class.best_estimator_
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/et_cv_data.csv',
                                                                       mode="w")
    Y_predict = et_clf.predict(X_test)
    sc = et_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/et.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Categorical Boosting
# 分类提升
def cat_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    cat = CatBoostClassifier()
    cat_param_grid1 = {
        'depth': [4, 6, 10],
        'learning_rate': [0.05, 0.1, 0.15],
        'l2_leaf_reg': [1, 4, 9],
        'iterations': [1200],
        'early_stopping_rounds': [1000],
        'loss_function': ['MultiClass'],
    }
    grid_search_class = GridSearchCV(cat, param_grid=cat_param_grid1, cv=3, scoring="roc_auc", n_jobs=-1, verbose=2)
    grid_search_class.fit(X_train, Y_train)
    cat_clf = grid_search_class.best_estimator_
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/cat_cv_data.csv',
                                                                       mode="w")
    Y_predict = cat_clf.predict(X_test)
    sc = cat_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/cat.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Light Gradient Boosting Machine
# 光线梯度提升机
def lgbm_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    lgbm = LGBMClassifier()
    lgbm_param_grid1 = {'num_leaves': range(5, 41, 5),
                        'max_depth': range(3, 22, 3),
                        'learning_rate': list(np.arange(0.01, 2, 0.1)),
                        }
    grid_search_class = GridSearchCV(lgbm, param_grid=lgbm_param_grid1, cv=3, scoring="roc_auc", n_jobs=-1, verbose=2)
    grid_search_class.fit(X_train, Y_train)
    lgbm_clf = grid_search_class.best_estimator_
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/lgbm_cv_data.csv',
                                                                       mode="w")
    Y_predict = lgbm_clf.predict(X_test)
    sc = lgbm_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/lgbm.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")


# Extreme Gradient Boosting
# 极端梯度提升
def xgb_fit(X_train, X_test, Y_train, Y_test):
    # Hyperparameter search
    xgb = XGBClassifier()
    xgb_param_grid1 = {
        'max_depth': range(3, 22, 3),
        'eta': list(np.arange(0.01, 2, 0.1)),
    }
    grid_search_class = GridSearchCV(xgb, param_grid=xgb_param_grid1, cv=3, scoring="roc_auc", n_jobs=-1, verbose=2)
    grid_search_class.fit(X_train, Y_train)
    xgb_clf = grid_search_class.best_estimator_
    a = [x for x in grid_search_class.cv_results_.values()]
    pd.DataFrame(a, index=grid_search_class.cv_results_.keys()).to_csv(f'./DataResource/Training_Result/xgb_cv_data.csv',
                                                                       mode="w")
    Y_predict = xgb_clf.predict(X_test)
    sc = xgb_clf.score(X_test, Y_test)
    cm = confusion_matrix(Y_test, Y_predict)
    ps = precision_score(Y_test, Y_predict)
    rs = recall_score(Y_test, Y_predict)
    f1 = f1_score(Y_test, Y_predict)
    with open(f"./DataResource/Training_Result/xgb.txt", mode="w") as f:
        f.write("best_params:" + "\n")
        f.write(str(grid_search_class.best_params_) + "\n")
        f.write("model_score:" + "\n")
        f.write(str(grid_search_class.best_score_) + "\n")

        f.write("best_score:" + "\n" + str(sc) + "\n")
        f.write("confusion_matrix:" + "\n" + str(cm) + "\n")
        f.write("precision_score:" + "\n" + str(ps) + "\n")
        f.write("recall_score:" + "\n" + str(rs) + "\n")
        f.write("f1_score:" + "\n" + str(f1) + "\n")
        f.write("========================================" + "\n")
