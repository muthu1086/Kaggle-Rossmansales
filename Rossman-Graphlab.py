import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import preprocessing
import graphlab as gl
from sklearn.base import TransformerMixin

print ("Loading CSV files")
train =pd.read_csv('/home/muthu/ipython/train.csv',low_memory=False)
test =pd.read_csv('/home/muthu/ipython/test.csv',low_memory=False)
store =pd.read_csv('/home/muthu/ipython/store.csv')

print ("Started Preprocessing")
#store = store.fillna(value=-9999)
train = pd.merge(train,store)
test = pd.merge(test,store)
train['year'] = train['Date'].apply(lambda x: int(x.split('-')[0]))
train['month'] = train['Date'].apply(lambda x: int(x.split('-')[1]))
train['Date'] = train['Date'].apply(lambda x: int(x.split('-')[2]))
test['year'] = test['Date'].apply(lambda x: int(x.split('-')[0]))
test['month'] = test['Date'].apply(lambda x: int(x.split('-')[1]))
test['Date'] = test['Date'].apply(lambda x: int(x.split('-')[2]))
cat_var = {'StateHoliday','StoreType','Assortment','PromoInterval'}


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

print ("Completed Preprocessing")
print ("Starting LabelEncoder")

for var in cat_var:
    lb = preprocessing.LabelEncoder()
    full_var_data = pd.concat((train[var],test[var]),axis=0).astype('str')
    lb.fit( full_var_data )
    train[var] = lb.transform(train[var].astype('str'))
    test[var] = lb.transform(test[var].astype('str'))

# Converting Pandas DataFrame to SFrame

train = gl.SFrame(data=train)
test = gl.SFrame(data=test)

# Model to predict the No.Of Customers which is missing in the test 

model = gl.random_forest_regression.create(train,target = 'Customers',features = ['Store','DayOfWeek','Date','Open','Promo','StateHoliday','SchoolHoliday','StoreType','Assortment','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2','Promo2SinceWeek','Promo2SinceYear', 'PromoInterval', 'year', 'month'],num_trees = 60)

predictions = model.predict(test)

# Adding Predicted customer column to test data 
test.add_column(predictions,name='Customers')

# Model to predict the Sales

model_sale = gl.random_forest_regression.create(train,target = 'Sales',features = ['Customers','Store','DayOfWeek','Date','Open','Promo','StateHoliday','SchoolHoliday','StoreType','Assortment','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2','Promo2SinceWeek','Promo2SinceYear', 'PromoInterval', 'year', 'month'],num_trees = 15)

predictions_sale = model_sale.predict(test)
test.add_column(predictions_sale,name='Sales')

# Extracting the result for submission

result = test.select_columns(['Id', 'Sales'])
result.save('ross-result1.csv', format='csv')

# Evaluation
results = model_sale.evaluate(train)
results

