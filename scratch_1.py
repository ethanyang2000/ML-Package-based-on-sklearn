from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier


class ModelCompare:
    __models = {}  # 私有变量，字典型。key是输入的别称，value是建立的模型
    predict = {}  # 公有变量 预测结果，key是别名，value是list
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
    __temp_estimator_in_cross_val_score=None
    # 选择输入的模型
    def __model_name(self, string, key):
        if string == 'RandomForestClassifier':
            self.__models[key] = RandomForestClassifier()
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

    # 默认参数进行训练
    def fit(self):
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(self.__x, self.__y,
                                                                                        test_size=0.2,
                                                                                        stratify=self.__y)
        self.__train_flag = 1
        for key in self.__models:
            self.__models[key].fit(self.__x_train, self.__y_train)

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
            pred = self.__models[key].predict(test)
            self.predict[key] = pred
        print(self.predict)

    # 打印accuracy 利用分裂的数据集，此分裂是在fit中产生的
    def showAccurancy(self):
        if self.__train_flag == 0:
            print('Models hasn\'t been trained')
            return
        for key in self.__models:
            accuracy = accuracy_score(y_true=self.__y_test, y_pred=self.__models[key].predict(self.__x_test))
            self.__scores[key] = accuracy
        print(self.__scores)

    def __getparamRF(self,key):
        self.__temp_estimator_in_cross_val_score=RandomForestClassifier(n_estimators=self.__best_params[key]['n_estimators'],
                                                                        max_features=self.__best_params[key]['max_features'])

    def __getparamLR(self,key):
        self.__temp_estimator_in_cross_val_score=LogisticRegression(penalty='l2',C=self.__best_params[key]['C'])

    def __getparamAda(self,key):
        self.__temp_estimator_in_cross_val_score=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=self.__best_params[key]['max_depth'],
                                                                    max_features=self.__best_params[key]['max_features']),
                                                                    n_estimators=self.__best_params[key]['n_estimators'])

    def __getparamKnn(self,key):
        if(self.__best_params[key]['weights']=='uniform'):
            self.__temp_estimator_in_cross_val_score=KNeighborsClassifier(weights='uniform',n_neighbors=self.__best_params[key]['n_neighbors'])
        else:
            self.__temp_estimator_in_cross_val_score=KNeighborsClassifier(weights='distance',n_neighbors=self.__best_params[key]['n_neighbors'],
                                                                          p=self.__best_params[key]['p'])

    def __getparamSvc(self,key):
        if self.__best_params[key]['kernel']=='rbf':
            self.__temp_estimator_in_cross_val_score=SVC(kernel='rbf',
                                                     C=self.__best_params[key]['C'],gamma=self.__best_params[key]['gamma'])
        else:
            self.__temp_estimator_in_cross_val_score=SVC(kernel='linear',C=self.__best_params[key]['C'])

    def __getparamXGB(self,key):
        params=self.__best_params[key]
        self.__temp_estimator_in_cross_val_score=XGBClassifier(n_estimators=self.__best_params[key]['n_estimators'],
                                                               gamma=params['gamma'],learning_rate=params['learning_rate'],
                                                               max_depth=params['max_depth'],min_child_weight=params['min_child_weight'],
                                                               subsample=params['subsample'],colsample_bytree=params['colsample_bytree'])
    def __getparamGB(self,key):
        self.__temp_estimator_in_cross_val_score=GradientBoostingClassifier(n_estimators=self.__best_params[key]['n_estimators'],
                                                                            max_depth=self.__best_params[key]['max_depth'],
                                                                            max_features=self.__best_params[key]['max_features'],
                                                                            learning_rate=self.__best_params[key]['learning_rate'])
    # 队训练好的模型在整个数据上做交叉验证
    def showCrossValScore(self):
        if len(self.__best_params) == 0:
            print("You should do GridSearch before checking your cross_val_score!")
            return
        for key in self.__models:
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
                self.__getparamGB(key)
            cross_val = cross_val_score(estimator=self.__temp_estimator_in_cross_val_score, X=self.__x, y=self.__y, cv=10, n_jobs=-1)
            self.__cross_val_scores[key] = cross_val
        print(self.__cross_val_scores)

    def __RFSearch(self, key):
        gs = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': range(1, 101, 10)},
                          scoring='roc_auc', cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestparam = gs.best_params_['n_estimators']
        gs = GridSearchCV(estimator=RandomForestClassifier(),
                          param_grid={'n_estimators': range(bestparam - 10, bestparam + 10, 2)}, scoring='roc_auc',
                          cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestparam = gs.best_params_['n_estimators']
        gs = GridSearchCV(estimator=RandomForestClassifier(),
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
        gs = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier()), param_grid={'n_estimators': range(1, 1000, 100)},
                          scoring='roc_auc', cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestparam = gs.best_params_['n_estimators']
        gs = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                          param_grid={"n_estimators": range(bestparam - 100, bestparam + 100, 10)}, scoring='roc_auc',
                          cv=10, n_jobs=-1)
        gs.fit(self.__x, self.__y)
        bestparam = gs.best_params_['n_estimators']
        for i in range(3, 14, 2):
            for j in range(1, 20, 2):
                tempmaxscore = \
                cross_val_score(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=i,
                                max_features=j),n_estimators=bestparam),X=self.__x, y=self.__y,
                                scoring='roc_auc', cv=10, n_jobs=-1).mean()
                if tempmaxscore > finalscore:
                    finalparam['max_depth'] = i
                    finalparam['max_features'] = j
                    finalscore = tempmaxscore
        finalparam['n_estimators'] = bestparam
        self.__best_scores[key] = finalscore
        self.__best_params[key] = finalparam
        self.__models[key] = AdaBoostClassifier(base_estimator=
            DecisionTreeClassifier(max_depth=finalparam['max_depth'], max_features=finalparam['max_features']),
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
        param_range_c = [ 0.001, 0.01, 0.1, 1, 10]
        param_gamma = [0.1, 0.2, 0.4, 0.6, 0.8, 1.6, 3.2, 6.4]
        #params = {'kernel': ['rbf'], 'C': [0.1, 0.2], 'gamma': [0.1, 0.2]}
        params=[{'kernel':['linear'],'C':param_range_c},{'kernel':['rbf'],'C':param_range_c,'gamma':param_gamma}]
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
        gs.fit(self.__x,self.__y)
        bestlr = gs.best_params_['learning_rate']
        gs=GridSearchCV(estimator=GradientBoostingClassifier(),
                        param_grid={"learning_rate":[bestlr],'n_estimators':[bestparam],'max_depth':range(3,14,2),
                                    'max_features':range(1,20,2)},cv=10,scoring='roc_auc',n_jobs=-1,refit=True)
        gs.fit(self.__x,self.__y)
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
        self.__models[key]=gs.best_estimator_
        self.__best_params[key]=gs.best_params_
        self.__best_scores[key]=gs.best_score_

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
        bestsub = gs.best_params_['subsample']
        bestcol = gs.best_params_['colsample_bytree']
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

    def OfficalVoting(self, inp, test):
        est = {}
        for key in inp:
            est[key] = self.__model_name(key)
        vc = VotingClassifier(estimators=est)
        vc.fit(self.__x, self.__y)
        print(cross_val_score(estimator=vc, x=self.__x, y=self.__y, cv=10, n_jobs=-1))
        print(vc.predict(test))
        return vc.predict(test)

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
        for i in range(len(inp)): prediction.append(est[i].predict(test))
        for i in range(len(test)):
            for j in range(len(inp)):
                temans += prediction[j][i]
            if (temans > len(inp) / 2):
                ans.append(1)
            else:
                ans.append(0)
            temans = 0
        print(ans)
        return ans
