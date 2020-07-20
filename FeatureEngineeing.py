import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, KernelPCA
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
class SBS():
    datadim = 0

    def __init__(self, estimator=RandomForestClassifier(), k_features=1, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, x, y):
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)
        dim = x_train.shape[1]
        self.datadim = dim
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(x_train, y_train, x_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(x_train, y_train, x_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def show(self):
        k_feat = [len(k) for k in self.subsets_]
        plt.plot(k_feat, self.scores_, marker='o')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of features')
        plt.grid()
        plt.show()

    def _calc_score(self, x_train, y_train, x_test, y_test, indices):
        self.estimator.fit(x_train.values[:, indices], y_train)
        y_pred = self.estimator.predict(x_test.values[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


class FeatureEngineering:
    __x = None       #输入的x
    __y = None          #输入的y
    __test = None
    __total = None
    __x_mms = None
    __x_ss = None
    __total_mms = None
    __total_ss = None
    __test_mms = None
    __test_ss = None
    __sbs_flag = 0
    x_dr = None #降维后数据
    test_dr = None
    # 初始化，加载数据集
    def __init__(self, x, y, test):
        self.__y = y
        self.__x = x
        self.__test = test
        self.__total = pd.concat([self.__test, self.__x])
        print('Data loaded successfully!')
    #检查数据中的缺省值
    def checkNull(self):
        print(self.__total.isnull().sum())
    #丢弃不需要的特征
    def dropCol(self, string_list):
        self.__total.drop(string_list, axis=1, inplace=True)
        for column in string_list:
            print(column+' has been dropped!')
    #丢弃不要的特征
    def dropRows(self, string_list):
        self.__total.drop(string_list, axis=0, inplace=True)
        for row in string_list:
            print('row'+str(row) + ' has been dropped!')
    #丢弃包含null值的行
    def dropNullRows(self, how='all', thresh=None, subset=None):
        self.__total = self.__total.dropna(axis=0, subset=subset, thresh=thresh, how=how)
        print('Null rows have been dropped')
    #插值
    def Imputer(self, strategy='mean', column_list=None, const=None):
        for column in column_list:
            if strategy == 'const':
                self.__total[column] = self.__total[column].fillna(const)
            if strategy == 'mean':
                self.__total[column] = self.__total[column].fillna(self.__total[column].mean())
            if strategy == 'mode':
                self.__total[column] = self.__total[column].fillna(self.__total[column].mode())
            if strategy == 'interpolate':
                self.__total[column] = self.__total[column].interpolate()
            print("values in " + column + ' has been imputed!')
    #将类别特征对应于数字
    #map_rules应为字典
    def mapping(self, column, map_rules):
        self.__total[column]=self.__total[column].map(map_rules)
        print('mapping succeed!')
    #对需要序数编码的特征编码
    def labelEncoder(self, column_list):
        for column in column_list:
            le = LabelEncoder()
            self.__total[column] = le.fit_transform(self.__total[column])
            print('encoder of ' + column + ' succeed!')
    # 独热编码
    def oneHotEncoder(self, column_list):
        for column in column_list:
            map = {i: column + str(i) for i in range(100)}
            oh = pd.get_dummies(self.__total[column])
            oh.drop(oh.columns.values.tolist()[0], axis=1, inplace=True)
            oh.rename(columns=map, inplace=True)
            self.__total.drop([column], axis=1, inplace=True)
            self.__total = pd.concat([self.__total, oh], axis=1)
            print("One hot encoder of " + column + ' has been completed!')
    # 进行归一化
    def mmScaler(self):
        mms = MinMaxScaler()
        self.__total_mms = mms.fit_transform(self.__total)
        print('MinMaxScaler has been completed!')
    # 进行标准化
    def sScaler(self):
        ss = StandardScaler()
        self.__total_ss = ss.fit_transform(self.__total)
        print('StandardScaler has been completed!')
    # 进行逆序数特征选择
    def SBS(self, estimator=RandomForestClassifier(), k_features=1, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.__sbs_flag = 1
        self.__sbs = SBS(estimator=estimator, k_features=k_features, scoring=scoring, test_size=test_size, random_state=random_state)
        self.__sbs.fit(self.__x, self.__y)
        self.__sbs.show()
        for i in range(len(self.__sbs.subsets_)):
            print('####The best subset of '+str(self.__sbs.datadim-i)+' features is ####')
            k=list(self.__sbs.subsets_[i])
            for l in k:
                print(self.__total.columns[l])

    def SBS_transform(self,n_feature):
        if(n_feature<self.__sbs.k_features):
            print('n_features that you input is too small!')
            return
        index=self.__sbs.datadim-n_feature
        temlist=list(self.__sbs.subsets_[index])
        collist=[]
        for i in temlist:
            collist.append(self.__total.columns[i])
        for column in list(self.__total.columns[1:]):
            if column not in collist:
                self.__total.drop([column], axis=1, inplace=True)
        print('sbs transform complete!')
    # 展示特征重要性
    def featureImportance(self):
        feat_labels=self.__total.columns[1:]
        forest=RandomForestClassifier(n_estimators=500,n_jobs=-1)
        forest.fit(self.__x,self.__y)
        importances=forest.feature_importances_
        indices=np.argsort(importances)[::-1]
        for f in range(self.__x.shape[1]):
            print('%2d) %-*s %f' % (f+1,30,feat_labels[indices[f]-1],importances[indices[f]-1]))
        plt.title('Feature Importance')
        plt.bar(range(self.__x.shape[1]),importances[indices],align='center')
        plt.xticks(range(self.__x.shape[1]),feat_labels,rotation=90)
        plt.xlim([-1,self.__x.shape[1]])
        plt.tight_layout()
        plt.show()

    #def plot_decision_region(self,classifier)

    def dimensionality_reduction(self, method='PCA',n_components=2,copy=False):
        if method == 'PCA':
            pca = PCA(n_components=n_components,copy=False)
            self.x_dr = pca.fit_transform(self.__x)
            self.test_dr = pca.transform(self.__test)
        elif method == 'LDA':
            lda = LDA(n_components=n_components,copy=False)
            self.x_dr = lda.fit_transform(self.__x)
            self.test_dr = lda.transform(self.__test)
        elif method == 'kPCA':
            kPCA = KernelPCA(n_components=n_components,copy=False,kernel='rbf',gamma=15)
            self.x_dr = kPCA.fit_transform(self.__x)
            self.test_dr = kPCA.transform(self.__test)
        return self.x_dr, self.test_dr



