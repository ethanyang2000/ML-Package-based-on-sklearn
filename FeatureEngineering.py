import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
from sklearn.decomposition import PCA, KernelPCA
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class SBS():
    datadim = 0

    def __init__(self, estimator=RandomForestClassifier(), k_features=1, scoring=accuracy_score, test_size=0.25,
                 random_state=1):
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


class FeatureEngineering:  # ！！！！！！！！！！输入后立刻将测试集和训练集的特征合并为total共同处理，在最后增加一个函数将total分裂还原并返回

    __x = None
    __y = None
    __test = None
    __total = None  # 将测试集和训练集的标签合并
    __x_mms = None  # x归一化
    __x_ss = None  # x标准化
    __total_mms = None  # total归一化
    __total_ss = None
    __test_mms = None  # 测试归一化
    __test_ss = None
    __sbs_flag = 0
    __total_dr = None
    
    # 初始化，加载数据集
    def __init__(self, x, y, test):
        self.__test = pd.DataFrame(test).copy(deep=True)
        self.__x = pd.DataFrame(x).copy(deep=True)
        self.__y = pd.DataFrame(y).copy(deep=True)
        self.__total = pd.concat([self.__x, self.__test])
        self.__x_id=self.__x.index
        self.__test_id=self.__test.index
        warnings.filterwarnings('ignore')
        print(self.__total.tail())
        print('Data loaded successfully!')

    def checkNull(self):
        print(self.__total.isnull().sum())
        return self.__total.isnull().sum().loc[self.__total.isnull().sum()>0].index

    def dropCol(self, string_list):
        self.__total.drop(string_list, axis=1, inplace=True)
        for column in string_list:
            print(column + ' has been dropped!')

    def dropRows(self, string_list):
        self.__total.drop(string_list, axis=0, inplace=True)
        for row in string_list:
            print('row' + str(row) + ' has been dropped!')

    def dropNullRows(self, how='all', thresh=None, subset=None):
        self.__total = self.__total.dropna(axis=0, subset=subset, thresh=thresh, how=how)
        print('Null rows have been dropped')

    def Imputer(self, strategy='mean', column_list=[], const=1):
        for column in column_list:
            print(column)
            if strategy == 'const':
                insert_value=const
            if strategy == 'mean':
                insert_value=self.__total[column].mean()
            if strategy == 'mode':
                insert_value=self.__total[column].mode()
            null_list=list(self.__total.loc[self.__total[column].isnull()].index)
            if(type(insert_value)==pd.Series):
                value_=insert_value[0]
            else: value_=insert_value
            for row in null_list:
                self.__total.loc[row,column]=value_
            print("values in " + column + ' has been imputed!')

    def mapping(self, column, map_rules):
        self.__total[column] = self.__total[column].map(map_rules)
        print('mapping succeed!')

    def labelEncoder(self, column_list):
        for column in column_list:
            true_values = self.__total.loc[self.__total[column].notnull()]
            true_id = self.__total.loc[self.__total[column].notnull()].index
            le = LabelEncoder()
            true_values[column] = le.fit_transform(true_values[column])
            for i in range(true_values.shape[0]):
                self.__total.loc[true_id[i],column] = true_values[column].values[i]
            print('encoder of ' + column + ' succeed!')

    def oneHotEncoder(self, column_list):
        for column in column_list:
            maps = {i: column + str(i) for i in range(100)}
            oh = pd.get_dummies(self.__total[column])
            oh.drop(oh.columns.values.tolist()[0], axis=1, inplace=True)
            oh.rename(columns=maps, inplace=True)
            self.__total.drop([column], axis=1, inplace=True)
            self.__total = pd.concat([self.__total, oh], axis=1)
            print("One hot encoder of " + column + ' has been completed!')

    def mmScaler(self):
        mms = MinMaxScaler()
        self.__total_mms = mms.fit_transform(self.__total)
        print('MinMaxScaler has been completed!')

    def sScaler(self):
        ss = StandardScaler()
        self.__total_ss = ss.fit_transform(self.__total)
        print('StandardScaler has been completed!')

    def SBS(self, estimator=RandomForestClassifier(), k_features=1, scoring=accuracy_score, test_size=0.25,
            random_state=1):
        self.__sbs_flag = 1
        self.__sbs = SBS(estimator=estimator, k_features=k_features, scoring=scoring, test_size=test_size,
                         random_state=random_state)
        self.__sbs.fit(self.__x, self.__y)
        self.__sbs.show()
        for i in range(len(self.__sbs.subsets_)):
            print('####The best subset of ' + str(self.__sbs.datadim - i) + ' features is ####')
            k = list(self.__sbs.subsets_[i])
            for l in k:
                print(self.__total.columns[l])

    def SBS_transform(self, n_feature):
        if n_feature < self.__sbs.k_features:
            print('n_features that you input is too small!')
            return
        index = self.__sbs.datadim - n_feature
        temlist = list(self.__sbs.subsets_[index])
        collist = []
        for i in temlist:
            collist.append(self.__total.columns[i])
        for column in list(self.__total.columns[1:]):
            if column not in collist:
                self.__total.drop([column], axis=1, inplace=True)
        print('sbs transform complete!')

    def featureImportance(self):
        feat_labels = self.__total.columns[1:]
        forest = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        forest.fit(self.__x, self.__y)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(self.__x.shape[1]):
            print('%2d) %-*s %f' % (f + 1, 30, feat_labels[indices[f] - 1], importances[indices[f] - 1]))
        plt.title('Feature Importance')
        plt.bar(range(self.__x.shape[1]), importances[indices], align='center')
        plt.xticks(range(self.__x.shape[1]), feat_labels, rotation=90)
        plt.xlim([-1, self.__x.shape[1]])
        plt.tight_layout()
        plt.show()

    def checkString(self):
        String_list = []
        temp = self.__total.iloc[:6, 0:]
        for i in range(temp.shape[1]):
            count = 0
            for j in range(1, 6):
                if type(temp.iloc[j, i]) == str:
                    count += 1
            if count > 0:
                String_list.append(list(temp.columns)[i])
        return String_list

    def __RF(self, column):
        data_total = self.__total.copy(deep=True)
        for col in list(self.__total.columns):
            if col == column:
                continue
            else:
                tem = self.__publicnum(list(data_total.loc[data_total[col].notnull()][col].values))
                index_tem = data_total.loc[data_total[col].isnull()].index.tolist()
                for i in index_tem:
                    data_total.loc[i,col] = tem
        Null_rows = data_total[data_total[column].isnull()]
        Null_rows_id = list(data_total.loc[data_total[column].isnull()].index)
        train_data_id = list(data_total.loc[data_total[column].notnull()].index)
        train_data = data_total.loc[data_total[column].notnull()]
        train_y = train_data[column]
        train_x = train_data.drop([column], axis=1)
        Null_rows.drop([column], axis=1, inplace=True)
        if self.__checkFloat(train_y):
            rfr = RandomForestRegressor(n_jobs=-1,n_estimators=50,max_depth=10,min_samples_leaf=5)
            rfr.fit(train_x, train_y)
            pred = rfr.predict(Null_rows)
        else:
            for temp_col in train_x.columns:
                if train_x[temp_col].dtype==float:
                    continue
                train_x[temp_col]=pd.to_numeric(train_x[temp_col], errors='raise').astype('int32')
            train_y=pd.to_numeric(train_y, errors='coerce').astype('int32')
            rfc = RandomForestClassifier(n_jobs=-1,n_estimators=50,max_depth=10,min_samples_leaf=5)
            rfc.fit(train_x, train_y)
            pred = rfc.predict(Null_rows)
        for i in range(train_data.shape[0]):
            self.__total.loc[train_data_id[i],column] = train_data[column].values[i]
        for i in range(Null_rows.shape[0]):
            temp_id = Null_rows_id[i]
            self.__total.loc[temp_id,column] = pred[i]
        print('RF Impute of ' + column + ' has been done!')

    def RFImpute(self):
        Null_list = self.__total.isnull().sum()
        Null_limit=self.__total.shape[0]*0.01
        Null_list = Null_list.loc[Null_list.values > Null_limit]
        Null_index = list(Null_list.sort_values(ascending=False).index)
        for col in Null_index:
            self.__RF(col)

    def __checkstring(self, data):
        String_list = []
        temp = data.iloc[:6, 0:]
        for i in range(temp.shape[1]):
            count = 0
            for j in range(1, 6):
                if type(temp.iloc[j, i]) == str:
                    count += 1
            if count > 0:
                String_list.append(list(temp.columns)[i])
        return String_list

    def __checkFloat(self, data):
        ans = False
        for i in range(1, 6):
            if type(data.values[i]) == 'float':
                ans = True
                break
        return ans

    '''
    def check(self,data,str):
        for col in data.columns:
            for row in range(len(data)):
                if data[col].values[row]==str:
                    print(col,row)
    '''

    def __publicnum(self, inp, d=0):
        dictnum = {}
        for i in range(len(inp)):
            if inp[i] == np.nan or inp[i] == None:
                continue
            if inp[i] in dictnum.keys():
                dictnum[inp[i]] += 1
            else:
                dictnum.setdefault(inp[i], 1)
        maxnum = 0
        maxkey = 0
        for k, v in dictnum.items():
            if v > maxnum:
                maxnum = v
                maxkey = k
        return maxkey

    def split_data(self):

        test = self.__total.iloc[self.__test_id[0]-1:,:]
        train = self.__total.iloc[0:self.__test_id[0]-1,:]

        test.set_index(self.__test.index)
        train.set_index(self.__x.index)

        return train, self.__y, test
    
    def display_explain_variance(self):
        # need to be standardized
        if self.__total_ss is None:
            self.sScaler()
        cov_mat = np.cov(self.__total_ss.T)
        eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
        tot = sum(eigen_vals)
        var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        plt.bar(np.arange(1,self.__total_ss.shape[1]+1),np.array(var_exp),alpha=0.5,align='center',label='individual explained variance')
        plt.step(np.arange(1,self.__total_ss.shape[1]+1),np.array(cum_var_exp),where='mid',label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('principal component index')
        plt.legend(loc='best')
        plt.show()

    def dimensionality_reduction(self, method='PCA', n_components=2, copy=False):
        if method == 'PCA':
            self.display_explain_variance()
            pca = PCA(n_components=n_components, copy=copy)
            self.__total_dr = pd.DataFrame(pca.fit_transform(self.__total_ss))
            
        elif method == 'LDA':
            lda = LDA(n_components=n_components, copy=copy)
            if self.__total_ss is None:
                self.sScaler()
            self.__total_dr = pd.DataFrame(lda.fit_transform(self.__total_ss))
        elif method == 'kPCA':
            if self.__total_ss is None:
                self.sScaler()
            kPCA = KernelPCA(n_components=n_components, copy=copy, kernel='rbf', gamma=15)
            self.__total_dr = pd.DataFrame(kPCA.fit_transform(self.__total_ss))
        print(method+' has been done!')

    def updateFromDR(self):
        self.__total=self.__total_dr
        print('results of DR has been updated!')
        
