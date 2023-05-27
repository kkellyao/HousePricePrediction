'''
1. Use pandas for data cleaning
2. Matplotlib for data visulaiztion
3. Sklearn for ML model
4. predict sale price
'''
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from matplotlib import pyplot as plt
from sklearn.inspection._permutation_importance import permutation_importance
from sklearn import preprocessing, svm, ensemble

import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)

############# read data into Pandas  #########
df = pd.read_csv('realtorGApg-1-6.csv')

############# Create Function to Display DataFrame Contents #####
def df_info(df):
    print(df.head(1))
    print(df.columns)
    print('(rows, columns)', df.shape)
    
df_info(df)
############# Drop unuse columns #############
df2 = df.drop(['web-scraper-order','web-scraper-start-url','properties','properties-href'], axis='columns')
df_info(df2)

####### Cleaning Up NUll value #####################
####### Display how many NULL in each columns ######
####### Drop rows which has null value #############
print(df2.isnull().sum())
df3 = df2.dropna()
print(df3.isnull().sum())


######## Create New Column 'zip' from  extract from address #######
df3['zip'] = df3['address'].str.extract(r'\b(\w+)$', expand=True)
df_info(df3)



######## create new column called 'price_per_sqft' #############
######## step1 remove '$'
df3['price']=df3['price'].str.replace('$','')
######## step2 remove ','
df3['price']=df3['price'].str.replace(',','')
df3['sqft']=df3['sqft'].str.replace(',','')
######## step3 convert price object type to float
df3['price']= pd.to_numeric(df3['price'], errors='coerce' )
df3['sqft']= pd.to_numeric(df3['sqft'], errors='coerce' )
df3['price_per_sqft']=df3['price']/df3['sqft']

######## Drop address column
df4=df3.drop(['address'], axis='columns')
print(df4)

######## Check there is no records sqft/br < 100 ####
print(df4[df4.sqft/df4.bed < 100].head())
'''
print(df4.shape)
df5 = df4[~(df4.sqft/df4.bed < 100)]
# Check there is no records sqft/br < 100
print(df5[df5.sqft/df5.bed < 100])
'''
####### Check price_per_sqft and remove extreme ######
####### high price beyond one Standard Deviation #####
print(df4.price_per_sqft.describe())

# Display to majorirty of property price per square feet
# to see normal distribution
#import matplotlib 
def plot_hist(column):
    matplotlib.rcParams['figure.figsize']=(5,5)
    plt.hist(column, rwidth=0.8)
    plt.xlabel("Price Per Square Feet")
    plt.ylabel("Count")
    plt.show()
    
plot_hist(df4.price_per_sqft)

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('zip'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
df5 = remove_pps_outliers(df4)
plot_hist(df5.price_per_sqft)#print(df5.shape)

# check price vs 2 or 3 bedroom using plot scatter chart
def plot_scatter_chart(df, zip):
    bdr2 = df[(df.zip==zip) & (df.bed==2)]
    bdr3 = df[(df.zip==zip) & (df.bed==3)]
    matplotlib.rcParams['figure.figsize']=(5,5)
    plt.scatter(bdr2.sqft, bdr2.price, color='blue', label='2 Bedroom', s=50)
    plt.scatter(bdr3.sqft, bdr3.price, marker='+', color='green', label='3 Bedroom', s=50)
    plt.xlabel('Total Squre Feet Area')
    plt.ylabel('Price')
    plt.title(zip)
    plt.legend()

plot_scatter_chart(df5, '30062')
plt.show()

######## Cleaning up 'bath' value has '+'
######## Find Unique name 
print(df5['bath'].unique())
df5['bath']=df5['bath'].str[:3]
# Make a new column call 'bath' which only hold number of bath
#df5['bath'] = df5['bath'].apply(lambda x: int(x.split('+')[0]))
print(df5['bath'].unique())

###### ML Model Building ########
###### convert bed, bath, zip objects type to number type
df5['bed']= pd.to_numeric(df5['bed'], errors='coerce' )
df5['bath']= pd.to_numeric(df5['bath'], errors='coerce' )
df5['zip']= pd.to_numeric(df5['zip'], errors='coerce' )

# X are all independent variables and y is prediction variable
X = df5.drop(['price', 'price_per_sqft'], axis='columns')
print(X.head())
y = df5.price
print(y.head())

####### X are all independent variables and ########
####### y is prediction variable #########
X = df5.drop(['price', 'price_per_sqft'], axis='columns')
y = df5.price


# split into train and test data and set random-state=5 in order to generate 
# result when each time run. choose 5 because score it highest.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=5)
# Use a linearRegression Model
from sklearn.linear_model import LinearRegression
# create LinearRegression Model
lr_model = LinearRegression()
# Training data into Model
lr_model.fit(X_train, y_train)

def feature_prem_importance(model,  X_test, y_test):
    # Display feature permulation importance measure the predictive value of a feature.
    # Evaluating how the  prediction error increase when  a feature is  not availabe. 
    result = permutation_importance(lr_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig,  ax =plt.subplots()
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T)
    ax.set_title("Premutation Importances (test set)")
    fig.tight_layout()
    plt.show()
# call display function
feature_prem_importance(lr_model, X_test, y_test)

# Evaluation model score
print(lr_model.score(X_test, y_test))

def predict_price(model, bed, bath, sqft, zip):
    x = np.zeros(len(X.columns))
    x[0] = bed
    x[1] = bath
    x[2] = sqft
    x[3] = zip
    return model.predict([x])
    
print(predict_price(lr_model, 4,3.5, 3765, 30062))

### We will try different parameters to come up with the best performance on 
### LinearRegression Model
### we use a K-fold cross-validation by spliting the data, fitting a model
### and computing the score 5 consecutive times
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=5) # 10, 5
scores = cross_val_score(LinearRegression(), X, y, cv=cv)
print("%0.2f accuracy with a standard  deviation  of %0.2f" % (scores.mean(), scores.std()))

# We will try couple of algorithms models with couple of different parameters  to  come
# up with the best optimal model. We are using Grid Search CV method
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
# Find the best parameter this is called hyper parameter tuning

def find_best_model_using_gridsearchcv(X, y):
    algos={
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random','cyclic']
            }
        },
        'decision_tree':{
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter' : ['best','random']
            }
        }
    }

    scores=[]
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model','best_score','best_params'])

clf = ensemble.GradientBoostingRegressor(n_estimators=14, max_depth=18, min_samples_split=2, learning_rate=0.1, loss='ls')
clf.fit(X_train, y_train)
print('After Gradient Boosting Accuracy Score: ', "{:.4f}".format(clf.score(X_test, y_test)))
print(find_best_model_using_gridsearchcv(X, y).head())
print(predict_price(clf, 4,3.5, 3765, 30062))