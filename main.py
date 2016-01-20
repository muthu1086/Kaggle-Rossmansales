import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import svm

print ("Loading CSV files")
train =pd.read_csv('/home/hadoop2/muthu/ipython/sales-train.csv',low_memory=False)
test =pd.read_csv('/home/hadoop2/muthu/ipython/sales-test.csv',low_memory=False)
store =pd.read_csv('/home/hadoop2/muthu/ipython/store.csv')


train = pd.merge(train,store)
test = pd.merge(test,store)
print ("Cleaning Data Started")

train['year'] = train['Date'].apply(lambda x: int(x.split('-')[0]))
train['month'] = train['Date'].apply(lambda x: int(x.split('-')[1]))
train['Date'] = train['Date'].apply(lambda x: int(x.split('-')[2]))
test['year'] = test['Date'].apply(lambda x: int(x.split('-')[0]))
test['month'] = test['Date'].apply(lambda x: int(x.split('-')[1]))
test['Date'] = test['Date'].apply(lambda x: int(x.split('-')[2]))

#test['Open'] = test['Open'].fillna(value=test['Open'].median())
#train['Open'] = train['Open'].fillna(value=train['Open'].median())
train['Promo2SinceWeek'].fillna(0, inplace=True)
test['Promo2SinceWeek'].fillna(0, inplace=True)
train['Promo2SinceYear'].fillna(0, inplace=True)
test['Promo2SinceYear'].fillna(0, inplace=True)
train['PromoInterval'].fillna(0, inplace=True)
test['PromoInterval'].fillna(0, inplace=True)

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


X = pd.DataFrame(train)
train = DataFrameImputer().fit_transform(X)

Y = pd.DataFrame(test)
test = DataFrameImputer().fit_transform(Y)

col = {'StateHoliday','StoreType','Assortment'}

def dummy(col,data):
	
	for a in col:
		dummy_train = pd.get_dummies(data[a],prefix=a)
		data = pd.concat([data,dummy_train],axis=1)
		data.drop(a,axis=1,inplace=True)
	return data

train = dummy(col,train)
test = dummy(col,test)
test['StateHoliday_b'] = 0
test[ 'StateHoliday_c'] = 0

print(train.columns)
print(test.columns)
print ("Cleaning Data Done")
features = {'Store', 'DayOfWeek', 'Date', 'Open', 'Promo','SchoolHoliday', 'CompetitionDistance', 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek','Promo2SinceYear','year', 'month', 'StateHoliday_0', 'StateHoliday_a','StateHoliday_b', 'StateHoliday_c', 'StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d', 'Assortment_a', 'Assortment_b', 'Assortment_c'}
'''
for a in train.columns:
    per = (train[a].isnull().sum()/train[a].count())*100
    count = train[a].isnull().sum()
    print(a,count,per)


for a in test.columns:
    per = (train[a].isnull().sum()/train[a].count())*100
    count = train[a].isnull().sum()
    print(a,count,per)
'''

print ("Started to generate data for model")
x_train = train[list(features)].values
print ("completed X_train")
y_train = train['Sales'].values
print ("completed Y_train")
x_test=test[list(features)].values
print ("completed X_test")
'''lr = LinearRegression()
lr.fit(x_train,y_train)
rec = lr.predict(x_test)
rf = RandomForestClassifier(n_estimators=10,n_jobs=1)
print ("Starting fit in the Model")
rf.fit(x_train, y_train)'''
#clf = svm.SVC(kernel='linear',verbose=True)
#print (clf)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
print (gnb)
print ("Training the Model")
gnb.fit(x_train,y_train)

print ("Model has been successfully generated")
print ("Applying Model on Data")
#rec= rf.predict(x_test)
rec= gnb.predict(x_test)
print ("Completed the prediction, generating results")
test["Sales"]=rec
test.to_csv('/home/hadoop2/muthu/Data-Analytics/Muthu/RossmanStoreSales/Solution/out-svc.csv',columns=['Id','Sales'],index=False)

