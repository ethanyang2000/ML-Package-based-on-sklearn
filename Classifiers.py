from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, plot_roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.base import clone

import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd


class Classifiers:
    # members
    __models = {}  # 私有变量，字典型。key是输入的别称，value是建立的模型
    __best_models = []
    predict_data = {}  # 公有变量 预测结果，key是别名，value是list
    __scores = {}  # 默认值的分数，采用accuracy
    __cross_val_scores = {}  # 交叉验证的分数
    __best_params = {}  # 调参后的参数值，value是一个字典，参数名：参数值
    __best_scores = {}  # 调参后的分数
    __x = None  # 输入的完整数据集
    __y = None
    __x_train = None
    __x_test = None
    __y_train = None
    __y_test = None  # 分裂后的数据集
    __train_flag = None  # 是否训练过
    __search_flag = None  # 是否调过参
    __temp_estimator_in_cross_val_score = None

    # methods
    def __model_name(self, string, key):
        wrong_flag = 0
        if string == 'RandomForestClassifier':
            self.__models[key] = RandomForestClassifier(n_jobs=-1)
        elif string == 'LogisticRegression':
            self.__models[key] = LogisticRegression(max_iter=1000000)
        elif string == 'AdaBoostClassifier':
            self.__models[key] = AdaBoostClassifier()
        elif string == 'KNeighborsClassifier':
            self.__models[key] = KNeighborsClassifier()
        elif string == 'SVC':
            self.__models[key] = SVC()
        elif string == 'XGBClassifier':
            self.__models[key] = XGBClassifier()
        elif string == 'GradientBoostingClassifier':
            self.__models[key] = GradientBoostingClassifier()
        else:
            print('WARNING!YOU HAVE A WRONG INPUT!')
            wrong_flag = 1
        if wrong_flag == 0:
            print(string + ' has been loaded as ' + key + ' successfully!')
        __train_flag = 0
        __search_flag = 0

    # 构造函数，接受一个字典作为选择的模型
    def __init__(self, inp):
        self.__input = inp  # 私有变量，存储别名和模型的关系
        for key, value in inp.items():
            self.__model_name(value, key)  # 私有变量model存储初始化的模型

    def loadData(self, x, y):
        self.__x = x
        self.__y = y
        print('Data has been loaded successfully!')

    # 默认参数进行训练
    def fit(self):
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(self.__x, self.__y,
                                                                                        test_size=0.2,
                                                                                        stratify=self.__y)
        self.__train_flag = 1
        for key in self.__models:
            self.__models[key].fit(self.__x_train, self.__y_train)
        print('Models have been trained with default params!')

    # 公有方法，对测试集做出预测，更新predict字典
    def predict(self, test):
        if self.__train_flag == 0:
            print("models has not been trained!")
            return
        if self.__search_flag == 0:
            print("Params has not been optimized!")
            for key in self.__models:
                self.__models[key].fit(self.__x, self.__y)  # 在整个训练集上训练
        for key in self.__models:
            self.predict_data[key] = self.__models[key].predict(test).tolist()
        print('prediction completed!')

    # 打印accuracy 利用分裂的数据集，此分裂是在fit中产生的
    def showAccurancy(self):
        if self.__train_flag == 0:
            print('Models hasn\'t been trained')
            return
        for key in self.__models:
            accuracy = accuracy_score(y_true=self.__y_test, y_pred=self.__models[key].predict(self.__x_test))
            self.__scores[key] = accuracy
            print('The accuracy of ' + key + ' is ' + str(accuracy))

    """
    def __getparamRF(self, key):
        self.__temp_estimator_in_cross_val_score = RandomForestClassifier(
            n_estimators=self.__best_params[key]['n_estimators'],
            max_features=self.__best_params[key]['max_features'],
            n_jobs=-1)

    def __getparamLR(self, key):
        self.__temp_estimator_in_cross_val_score = LogisticRegression(penalty='l2', C=self.__best_params[key]['C'])

    def __getparamAda(self, key):
        self.__temp_estimator_in_cross_val_score = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=self.__best_params[key]['max_depth'],
                                                  max_features=self.__best_params[key]['max_features']),
            n_estimators=self.__best_params[key]['n_estimators'])

    def __getparamKnn(self, key):
        if self.__best_params[key]['weights'] == 'uniform':
            self.__temp_estimator_in_cross_val_score = KNeighborsClassifier(weights='uniform',
                                                                            n_neighbors=self.__best_params[key][
                                                                                'n_neighbors'])
        else:
            self.__temp_estimator_in_cross_val_score = KNeighborsClassifier(weights='distance',
                                                                            n_neighbors=self.__best_params[key][
                                                                                'n_neighbors'],
                                                                            p=self.__best_params[key]['p'])

    def __getparamSvc(self, key):
        if self.__best_params[key]['kernel'] == 'rbf':
            self.__temp_estimator_in_cross_val_score = SVC(kernel='rbf',
                                                           C=self.__best_params[key]['C'],
                                                           gamma=self.__best_params[key]['gamma'])
        else:
            self.__temp_estimator_in_cross_val_score = SVC(kernel='linear', C=self.__best_params[key]['C'])

    def __getparamXGB(self, key):
        params = self.__best_params[key]
        self.__temp_estimator_in_cross_val_score = XGBClassifier(n_estimators=self.__best_params[key]['n_estimators'],
                                                                 gamma=params['gamma'],
                                                                 learning_rate=params['learning_rate'],
                                                                 max_depth=params['max_depth'],
                                                                 min_child_weight=params['min_child_weight'],
                                                                 subsample=params['subsample'],
                                                                 colsample_bytree=params['colsample_bytree'])

    def __getparamGB(self, key):
        self.__temp_estimator_in_cross_val_score = GradientBoostingClassifier(
            n_estimators=self.__best_params[key]['n_estimators'],
            max_depth=self.__best_params[key]['max_depth'],
            max_features=self.__best_params[key]['max_features'],
            learning_rate=self.__best_params[key]['learning_rate'])
    """

    def showCrossValScore(self):
        if len(self.__best_params) == 0:
            print("You should do GridSearch before checking your cross_val_score!")
            return
        for key in self.__models:
            """
            if self.__input[key] == 'RandomForestClassifier':
                self.__getparamRF(key)
            elif self.__input[key] == 'LogisticRegression':
                self.__getparamLR(key)
            elif self.__input[key] == 'AdaBoostClassifier':
                self.__getparamAda(key)
            elif self.__input[key] == 'KNeighborsClassifier':
                self.__getparamKnn(key)
            elif self.__input[key] == 'SVC':
                self.__getparamSvc(key)
            elif self.__input[key] == 'XGBClassifier':
                self.__getparamXGB(key)
            elif self.__input[key] == 'GradientBoostingClassifier':
                self.__getparamGB(key)"""
            temp = clone(self.__models[key])
            cross_val = cross_val_score(estimator=temp, X=self.__x, y=self.__y,
                                        cv=10, n_jobs=-1)
            self.__cross_val_scores[key] = cross_val
            print('The cross val score of ' + key + ' is ' + str(cross_val))

    def __RFSearch(self, key):
        gs = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1), param_grid={'n_estimators': range(1, 101, 10)},
                          scoring='roc_auc', cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestparam = gs.best_params_['n_estimators']
        gs = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),
                          param_grid={'n_estimators': range(bestparam - 10, bestparam + 10, 2)}, scoring='roc_auc',
                          cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestparam = gs.best_params_['n_estimators']
        gs = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),
                          param_grid={'n_estimators': [bestparam], 'max_features': range(1, 11)}, scoring='roc_auc',
                          cv=10, n_jobs=-1, refit=True)
        gs.fit(self.__x, self.__y)
        self.__best_scores[key] = gs.best_score_
        self.__best_params[key] = gs.best_params_
        self.__models[key] = gs.best_estimator_

    def __LRSearch(self, key):
        params = {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        gs = GridSearchCV(estimator=LogisticRegression(), scoring='roc_auc', cv=10, n_jobs=-1, param_grid=params,
                          refit=True)
        gs.fit(self.__x, self.__y)
        self.__best_scores[key] = gs.best_score_
        self.__best_params[key] = gs.best_params_
        self.__models[key] = gs.best_estimator_

    def __adaSearch(self, key):
        finalparam = {}
        finalscore = 0
        gs = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                          param_grid={'n_estimators': range(1, 1000, 100)},
                          scoring='roc_auc', cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestparam = gs.best_params_['n_estimators']
        if bestparam > 100:
            gs = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                              param_grid={"n_estimators": range(bestparam - 100, bestparam + 100, 10)},
                              scoring='roc_auc',
                              cv=10, n_jobs=-1)
        else:
            gs = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                              param_grid={"n_estimators": range(1, bestparam + 100, 10)},
                              scoring='roc_auc',
                              cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestparam = gs.best_params_['n_estimators']
        for i in range(1, 14, 2):
            for j in range(1, 20, 2):
                tempmaxscore = \
                    cross_val_score(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=i,
                                                                                                       max_features=j),
                                                                 n_estimators=bestparam), X=self.__x, y=self.__y,
                                    scoring='roc_auc', cv=10, n_jobs=-1).mean()
                if tempmaxscore > finalscore:
                    finalparam['max_depth'] = i
                    finalparam['max_features'] = j
                    finalscore = tempmaxscore
        finalparam['n_estimators'] = bestparam
        self.__best_scores[key] = finalscore
        self.__best_params[key] = finalparam
        self.__models[key] = AdaBoostClassifier(base_estimator= \
                                                    DecisionTreeClassifier(max_depth=finalparam['max_depth'],
                                                                           max_features=finalparam['max_features']),
                                                n_estimators=bestparam)
        self.__models[key].fit(self.__x, self.__y)

    def __knnSearch(self, key):
        params = [{'weights': ['uniform'], 'n_neighbors': range(1, 11)},
                  {'weights': ['distance'], 'n_neighbors': range(1, 11), 'p': range(1, 6)}]
        gs = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, scoring='roc_auc', cv=10, n_jobs=-1,
                          refit=True)
        gs.fit(self.__x, self.__y)
        self.__best_scores[key] = gs.best_score_
        self.__best_params[key] = gs.best_params_
        self.__models[key] = gs.best_estimator_

    def __svcSearch(self, key):
        param_range_c = [0.1, 0.5, 1, 3, 7, 10, 20]
        param_gamma = [0.1, 0.2, 0.4, 0.6, 0.8, 1.6, 3.2, 6.4]
        # params = {'kernel': ['rbf'], 'C': [0.1, 0.2], 'gamma': [0.1, 0.2]}
        params = [{'kernel': ['linear'], 'C': param_range_c},
                  {'kernel': ['rbf'], 'C': param_range_c, 'gamma': param_gamma}]
        gs = GridSearchCV(estimator=SVC(), param_grid=params, n_jobs=-1, cv=10, scoring='roc_auc', refit=True)
        gs.fit(self.__x, self.__y)
        self.__best_scores[key] = gs.best_score_
        self.__best_params[key] = gs.best_params_
        self.__models[key] = gs.best_estimator_

    def __GBSearch(self, key):
        p = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
        gs = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid={'n_estimators': range(1, 101, 10)},
                          n_jobs=-1, cv=10, scoring='roc_auc')
        gs.fit(self.__x, self.__y)
        bestparam = gs.best_params_['n_estimators']
        gs = GridSearchCV(estimator=GradientBoostingClassifier(),
                          param_grid={'n_estimators': range(bestparam - 10, bestparam + 10, 2)}, cv=10,
                          scoring='roc_auc', n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestparam = gs.best_params_['n_estimators']
        gs = GridSearchCV(estimator=GradientBoostingClassifier(),
                          param_grid={'learning_rate': p, 'n_estimators': [bestparam]}, cv=10, scoring='roc_auc',
                          n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestlr = gs.best_params_['learning_rate']
        gs = GridSearchCV(estimator=GradientBoostingClassifier(),
                          param_grid={"learning_rate": [bestlr], 'n_estimators': [bestparam],
                                      'max_depth': range(3, 14, 2),
                                      'max_features': range(1, 20, 2)}, cv=10, scoring='roc_auc', n_jobs=-1, refit=True)
        gs.fit(self.__x, self.__y)
        """finalparam = {}
        finalparam['learning_rate'] = bestlr
        finalparam['n_estimators'] = bestparam
        finalscore = 0
        for i in range(3, 14, 2):
            for j in range(1, 20, 2):
                tempscore = \
                cross_val_score(estimator=GradientBoostingClassifier(), X=self.__x, y=self.__y,
                                fit_params={"base_estimator":DecisionTreeClassifier(max_depth=i,max_features=j),
                                            'n_estimators':bestparam,"learning_rate":bestlr},
                                scoring='roc_auc', cv=10, n_jobs=-1).mean()
                if (tempscore > finalscore):
                    finalparam['max_depth'] = i
                    finalparam['max_features'] = j
                    finalscore = tempscore
        self.__best_scores[key] = finalscore
        self.__best_params[key] = finalparam
        self.__models[key] = GradientBoostingClassifier(
            DecisionTreeClassifier(max_depth=finalparam['max_depth'], max_features=finalparam['max_features']),
            n_estimators=bestparam, learning_rate=bestlr)
        self.__models[key].fit(self.__x, self.__y)"""
        self.__models[key] = gs.best_estimator_
        self.__best_params[key] = gs.best_params_
        self.__best_scores[key] = gs.best_score_

    def __XGBSearch(self, key):
        p = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
        gs = GridSearchCV(estimator=XGBClassifier(),
                          param_grid={'learning_rate': p, 'n_estimators': range(1, 1000, 100)}, cv=10, n_jobs=-1,
                          scoring='roc_auc')
        gs.fit(self.__x, self.__y)
        bestlr = gs.best_params_['learning_rate']
        best_n = gs.best_params_['n_estimators']
        gs = GridSearchCV(estimator=XGBClassifier(),
                          param_grid={'learning_rate': [bestlr], 'n_estimators': range(best_n - 100, best_n + 100, 10)},
                          scoring='roc_auc', cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        best_n = gs.best_params_['n_estimators']
        gs = GridSearchCV(estimator=XGBClassifier(),
                          param_grid={'max_depth': range(3, 10, 2), 'min_child_weight': range(1, 6, 2),
                                      'learning_rate': [bestlr], 'n_estimators': [best_n]}, scoring='roc_auc', cv=10,
                          n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestdep = gs.best_params_['max_depth']
        bestwei = gs.best_params_['min_child_weight']
        gs = GridSearchCV(estimator=XGBClassifier(),
                          param_grid={'gamma': [i / 10.0 for i in range(0, 5)], 'max_depth': [bestdep],
                                      'learning_rate': [bestlr], 'n_estimators': [best_n],
                                      'min_child_weight': [bestwei]}, scoring='roc_auc', cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestgamma = gs.best_params_['gamma']
        gs = GridSearchCV(estimator=XGBClassifier(), param_grid={'subsample': [i / 10.0 for i in range(5, 10)],
                                                                 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
                                                                 'gamma': [bestgamma], 'max_depth': [bestdep],
                                                                 'learning_rate': [bestlr], 'n_estimators': [best_n],
                                                                 'min_child_weight': [bestwei]}, scoring='roc_auc',
                          cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestsub = gs.best_params_['subsample']
        bestcol = gs.best_params_['colsample_bytree']
        gs = GridSearchCV(estimator=XGBClassifier(), param_grid={
            'subsample': [i / 100.0 for i in range(int(bestsub * 100) - 10, int(bestsub * 100) + 10, 5)],
            'colsample_bytree': [i / 100.0 for i in range(int(bestcol * 100) - 10, int(bestcol * 100) + 10, 5)],
            'gamma': [bestgamma], 'max_depth': [bestdep], 'learning_rate': [bestlr], 'n_estimators': [best_n],
            'min_child_weight': [bestwei]}, scoring='roc_auc', cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        self.__best_scores[key] = gs.best_score_
        self.__best_params[key] = gs.best_params_
        self.__models[key] = gs.best_estimator_

    def GridSearch(self):
        for key in self.__models:
            if self.__input[key] == 'RandomForestClassifier':
                self.__RFSearch(key)
            elif self.__input[key] == 'LogisticRegression':
                self.__LRSearch(key)
            elif self.__input[key] == 'AdaBoostClassifier':
                self.__adaSearch(key)
            elif self.__input[key] == 'KNeighborsClassifier':
                self.__knnSearch(key)
            elif self.__input[key] == 'SVC':
                self.__svcSearch(key)
            elif self.__input[key] == 'GradientBoostingClassifier':
                self.__GBSearch(key)
            elif self.__input[key] == 'XGBClassifier':
                self.__XGBSearch(key)
            print('params of ' + key + ' has been optimized!')
            self.__best_models.append(clone(self.__models[key]))

    def HardVoting(self, inp, test):
        est = []
        for key in inp:
            if key in list(self.__models.keys()):
                est.append(self.__models[key])
            elif key in list(self.__input.values()):
                new_dic = {k: v for v, k in self.__input.items()}
                est.append(self.__models[new_dic[key]])
        ans = []
        prediction = []
        temans = 0
        for i in range(len(inp)):
            prediction.append(est[i].predict(test))
        for i in range(len(test)):
            for j in range(len(inp)):
                temans += prediction[j][i]
            if temans > len(inp) / 2:
                ans.append(1)
            else:
                ans.append(0)
            temans = 0
        print('predictions of hard voting has been completed!')
        return ans

    # IMPORTANT INFO #
    # !!! X, y are in dataframe form not ndarray!!!
    # *** estimators should be a fitted model ***
    # cv should be a cross_validate model which is set default value
    """
    def get_best_models(self):

        for model in self.__models.keys():
            if self.__input[model] == 'RandomForestClassifier':
                newmodel = RandomForestClassifier(n_estimators=self.__best_params[model]['n_estimators'],
                                                  max_features=self.__best_params[model]['max_features'])
                self.__best_models.append(newmodel)
            elif self.__input[model] == 'LogisticRegression':
                newmodel = LogisticRegression(penalty='l2', C=self.__best_params[model]['C'])
                self.__best_models.append(newmodel)
            elif self.__input[model] == 'AdaBoostClassifier':
                newmodel = AdaBoostClassifier(
                    base_estimator=DecisionTreeClassifier(max_depth=self.__best_params[model]['max_depth'],
                                                          max_features=self.__best_params[model]['max_features']),
                    n_estimators=self.__best_params[model]['n_estimators'])
                self.__best_models.append(newmodel)
            elif self.__input[model] == 'KNeighborsClassifier':
                if self.__best_params[model]['weights'] == 'uniform':
                    newmodel = KNeighborsClassifier(weights='uniform',
                                                    n_neighbors=self.__best_params[model]['n_neighbors'])
                    self.__best_models.append(newmodel)
                else:
                    newmodel = KNeighborsClassifier(weights='distance',
                                                    n_neighbors=self.__best_params[model][
                                                        'n_neighbors'],
                                                    p=self.__best_params[model]['p'])
                    self.__best_models.append(newmodel)
            elif self.__input[model] == 'SVC':
                if self.__best_params[model]['kernel'] == 'rbf':
                    newmodel = SVC(kernel='rbf', C=self.__best_params[model]['C'],
                                   gamma=self.__best_params[model]['gamma'])
                    self.__best_models.append(newmodel)
                else:
                    newmodel = SVC(kernel='linear', C=self.__best_params[model]['C'])
                    self.__best_models.append(newmodel)
            elif self.__input[model] == 'XGBClassifier':
                params = self.__best_params[model]
                newmodel = XGBClassifier(n_estimators=self.__best_params[model]['n_estimators'],
                                         gamma=params['gamma'], learning_rate=params['learning_rate'],
                                         max_depth=params['max_depth'], min_child_weight=params['min_child_weight'],
                                         subsample=params['subsample'], colsample_bytree=params['colsample_bytree'])
                self.__best_models.append(newmodel)
            elif self.__input[model] == 'GradientBoostingClassifier':
                newmodel = GradientBoostingClassifier(n_estimators=self.__best_params[model]['n_estimators'],
                                                      max_depth=self.__best_params[model]['max_depth'],
                                                      max_features=self.__best_params[model]['max_features'],
                                                      learning_rate=self.__best_params[model]['learning_rate'])
                self.__best_models.append(newmodel)
    """

    def display_model_accuracy(self):
        ### params #####
        #   X: data, dataframe
        #   y: lable, dataframe, shape(n,) if result is strange, try shape(n,1)
        #   estimators: list contains different est.
        #   cv: k-fold cross-validation model, should be set params. StratifiedKFold(10, shuffle=True, random_state=123)
        #   sort: sort or not, default is True
        #   packages: sklearn.model_selection, pandas, numpy,
        #                                                      StratifiedKFold, cross_validate
        #
        ### output ####
        #   no return
        #   output a colorful table, with a sorted order
        #######################################################
        X = self.__x
        y = self.__y
        estimators = self.__best_models
        cv = StratifiedKFold(10, shuffle=True, random_state=123)
        sort = True
        #######################################################

        model_table = pd.DataFrame()
        row_index = 0

        for est in estimators:
            MLA_name = est.__class__.__name__
            model_table.loc[row_index, 'Model Name'] = MLA_name

            cv_results = cross_validate(
                est,
                X,
                y,
                cv=cv,
                scoring='accuracy',
                return_train_score=True,
                n_jobs=-1
            )
            model_table.loc[row_index, 'Train Accuracy Mean'] = cv_results[
                'train_score'].mean()
            model_table.loc[row_index, 'Test Accuracy Mean'] = cv_results[
                'test_score'].mean()
            model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
            model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

            row_index += 1
        if sort:
            model_table.sort_values(by=['Test Accuracy Mean'], ascending=False, inplace=True)

        # display(model_table.style.background_gradient(cmap='summer_r'))

        fig, ax = plt.subplots(figsize=(15, 9))

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=model_table.values,
                 colLabels=model_table.columns,
                 loc='center',
                 # rowLabels=model_table.index,
                 cellLoc='center',
                 colLoc='center')

        fig.tight_layout()

        plt.show()

    def display_feature_importance(self):
        # params
        # bins: maxium bins on x axis
        # packages:
        # import seaborn as sns
        # import math
        # from matplotlib.ticker import MaxNLocator
        # output: barplot
        ###################################
        ax_num = 0
        va_est = []
        estimators = self.__best_models
        X = self.__x
        y = self.__y
        bins = 12
        ####################################
        for e in estimators:
            e.fit(X, y)

            try:
                if e.feature_importances_ is not None:
                    ax_num += 1
                    MLA_name = e.__class__.__name__
                    va_est.append(MLA_name)
            except:
                estimators.remove(e)

        for e in estimators:
            if e.__class__.__name__ in va_est:
                continue
            else:
                estimators.remove(e)

        if len(va_est) != 0:
            ncolumn = math.ceil(ax_num // 2)

            fig, axes = plt.subplots(ncolumn, 2, figsize=(20, 14))
            axes = axes.flatten()

            for ax, estimator in zip(axes, estimators):
                estimator.fit(X, y)

                feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importances_, X.columns)),
                                           columns=['Value', 'Feature'])

                sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False), ax=ax,
                            palette='plasma')

                plt.title('Features')
                plt.tight_layout()
                ax.set(title=f'{estimator.__class__.__name__} Feature Impotances')
                ax.xaxis.set_major_locator(MaxNLocator(nbins=bins))

            plt.show()
        else:
            print('No estimator has "feature_importances_" attribute')

    def display_roc(self):
        # params ##
        # just like the funtion 'model_accuracy'
        # package
        # import matplotlib.pyplot as plt
        # numpy for the 'interp' function
        # from sklearn.metrics import plot_roc_curve
        # copy and rewrite from kaggler: Ertuğrul Demir, https://www.kaggle.com/datafan07
        #########################################################
        X = self.__x
        y = self.__y
        estimators = self.__best_models
        cv = StratifiedKFold(10, shuffle=True, random_state=123)
        #########################################################
        fig, axes = plt.subplots(math.ceil(len(estimators) / 2),
                                 2,
                                 figsize=(25, 50))
        axes = axes.flatten()

        for ax, estimator in zip(axes, estimators):
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for i, (train, test) in enumerate(cv.split(X, y)):
                estimator.fit(X.loc[train], y.loc[train])
                viz = plot_roc_curve(estimator,
                                     X.loc[test],
                                     y.loc[test],
                                     name='ROC fold {}'.format(i),
                                     alpha=0.3,
                                     lw=1,
                                     ax=ax)
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            ax.plot([0, 1], [0, 1],
                    linestyle='--',
                    lw=2,
                    color='r',
                    label='Chance',
                    alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr,
                    mean_tpr,
                    color='b',
                    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' %
                          (mean_auc, std_auc),
                    lw=2,
                    alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr,
                            tprs_lower,
                            tprs_upper,
                            color='grey',
                            alpha=.2,
                            label=r'$\pm$ 1 std. dev.')
            ax.set(xlim=[-0.02, 1.02],
                   ylim=[-0.02, 1.02],
                   title=f'{estimator.__class__.__name__} ROC')
            ax.legend(loc='best', prop={'size': 18})
        plt.show()

    def display_learning_curve(self):
        # ## params ylim: limits on y axis n_jobs: how many cores to compute train_sizes: need to be used in
        # learning_curve function,train_sizes : array-like, shape (n_ticks,), dtype float or int Relative or absolute
        # numbers of training examples that will be used to generate the learning curve. If the dtype is float,
        # it is regarded as a fraction of the maximum size of the training set (that is determined by the selected
        # validation method), i.e. it has to be within (0, 1]. Otherwise it is interpreted as absolute sizes of the
        # training sets. Note that for classification the number of samples usually have to be big enough to contain
        # at least one sample from each class. (default: np.linspace(0.1, 1.0, 5)) packages: import math from
        # matplotlib.ticker import MaxNLocator import matplotlib.pyplot as plt from sklearn.model_selection import
        # learning_curve totolly copy from kaggler: Ertuğrul Demir, https://www.kaggle.com/datafan07
        #
        # output: figures
        ######################################
        X = self.__x
        y = self.__y
        estimators = self.__best_models
        ylim = None
        cv = StratifiedKFold(10, shuffle=True, random_state=123)
        n_jobs = None
        train_sizes = np.linspace(.1, 1.0, 5)
        ######################################
        fig, axes = plt.subplots(math.ceil(len(estimators) / 2),
                                 2,
                                 figsize=(25, 50))
        axes = axes.flatten()

        for ax, estimator in zip(axes, estimators):

            ax.set_title(f'{estimator.__class__.__name__} Learning Curve')
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_xlabel('Training examples')
            ax.set_ylabel('Score')
            train_sizes, train_scores, test_scores, fit_times, _ = \
                learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                               train_sizes=train_sizes,
                               return_times=True)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            # Plot learning curve

            ax.fill_between(train_sizes,
                            train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std,
                            alpha=0.1,
                            color='r')
            ax.fill_between(train_sizes,
                            test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std,
                            alpha=0.1,
                            color='g')
            ax.plot(train_sizes,
                    train_scores_mean,
                    'o-',
                    color='r',
                    label='Training score')
            ax.plot(train_sizes,
                    test_scores_mean,
                    'o-',
                    color='g',
                    label='Cross-validation score')
            ax.legend(loc='best')
            ax.yaxis.set_major_locator(MaxNLocator(nbins=24))

        plt.show()

    def display_confusion_matrix(self):
        # params
        # testsize is used to split
        # ***** this estimator is expected to be with best parameters *****
        #
        # packages:
        # from sklearn.metrics import confusion_matrix
        # import matplotlib
        # from sklearn.model_selection import train_test_split
        ################################
        X = self.__x
        y = self.__y
        estimators = self.__best_models
        testsize = 0.25
        ################################
        fig, axes = plt.subplots(math.ceil(len(estimators) / 2), 2, figsize=(25, 50))

        axes = axes.flatten()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=123, stratify=y)

        for ax, estimator in zip(axes, estimators):

            estimator.fit(X_train, y_train)
            y_predicted = estimator.predict(X_test)
            confus_matrix = confusion_matrix(y_true=y_test, y_pred=y_predicted)
            ax.matshow(confus_matrix,
                       # cmap=plt.cm.Blues,
                       alpha=0.3)
            for i in range(confus_matrix.shape[0]):
                for j in range(confus_matrix.shape[1]):
                    ax.text(x=j, y=i, s=confus_matrix[i, j], fontsize=30, va='center', ha='center')
            plt.xlabel('predicted label')
            plt.ylabel('true label')

        plt.show()
