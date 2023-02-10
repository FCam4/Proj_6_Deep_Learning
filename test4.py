import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingRegressor

from sklearn.metrics import mean_squared_error


# Feature Scalling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

test = pd.read_csv('C:\\Users\\sebas\Downloads\\archive\\Google_Playstore.csv')


df_pstore_ml = test.dropna(subset=['App Name','Rating', 'Rating Count', 'Installs', 'Minimum Installs', 'Maximum Installs', 'Size', 'Developer Id'])
df_pstore_ml = pd.get_dummies(df_pstore_ml, columns=['Category'])

condition = df_pstore_ml[(df_pstore_ml['Size'] == 'Varies with device')].index
df_pstore_ml.drop(condition , inplace=True)

substring = "M"
def turn_M_to_bite (value):
  substring_m = "M"
  substring_k = "k"
  substring_g = "G"
  if substring_m in value:
    value = value.replace(substring_m, "")
    value = value.replace(",", ".")
    value = float(value)
    value = value*1000000
  elif substring_k in value: 
    value = value.replace(substring_k, "")
    value = value.replace(",", ".")
    # print(value)
    value = float(value)
    value = value*1000
  elif substring_g in value: 
    value = value.replace(substring_g, "")
    value = value.replace(",", ".")
    # print(value)
    value = float(value)
    value = value*1000000000
  else:
    return value
  return value

df_pstore_ml['Size'] = df_pstore_ml['Size'].apply(turn_M_to_bite)

X = df_pstore_ml[['Rating Count', 'Maximum Installs', 'Ad Supported', 'Size', 'Category_Action', 'Category_Adventure',
       'Category_Arcade', 'Category_Art & Design', 'Category_Auto & Vehicles',
       'Category_Beauty', 'Category_Board', 'Category_Books & Reference',
       'Category_Business', 'Category_Card', 'Category_Casino',
       'Category_Casual', 'Category_Comics', 'Category_Communication',
       'Category_Dating', 'Category_Education', 'Category_Educational',
       'Category_Entertainment', 'Category_Events', 'Category_Finance',
       'Category_Food & Drink', 'Category_Health & Fitness',
       'Category_House & Home', 'Category_Libraries & Demo',
       'Category_Lifestyle', 'Category_Maps & Navigation', 'Category_Medical',
       'Category_Music', 'Category_Music & Audio', 'Category_News & Magazines',
       'Category_Parenting', 'Category_Personalization',
       'Category_Photography', 'Category_Productivity', 'Category_Puzzle',
       'Category_Racing', 'Category_Role Playing', 'Category_Shopping',
       'Category_Simulation', 'Category_Social', 'Category_Sports',
       'Category_Strategy', 'Category_Tools', 'Category_Travel & Local',
       'Category_Trivia', 'Category_Video Players & Editors',
       'Category_Weather', 'Category_Word']]
y = df_pstore_ml['Rating']


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state=19)

reg1 = DecisionTreeRegressor()
reg2 = GradientBoostingRegressor()
meta_model = GradientBoostingRegressor()

# Create voting regressor
reg_stack = StackingRegressor(regressors=[reg1, reg2], meta_regressor=meta_model, use_features_in_secondary=False)

# Fit and predict with the models and ensemble
# algorithms_list = [reg1, reg2, reg3, voting_ens]
# algorithms_list = list(voting_ens)
# def voting_regressor_func(algorithm):
  # print(algorithm.__class__.__name__)
reg_stack.fit(X_train, y_train)
print("Training score", reg_stack.score(X_train, y_train))
print("Test score", reg_stack.score(X_test, y_test))

y_pred = reg_stack.predict(X_test)
print('StackingRegressor:')
print('The mean squared error for the StackingRegressor is:', mean_squared_error(y_test, y_pred))

# reg1 = DecisionTreeRegressor(max_depth=12)
# reg1.fit(X_train, y_train)
# print("Training score", reg1.score(X_train, y_train))
# print("Test score", reg1.score(X_test, y_test))

# y_pred = reg1.predict(X_test)
# print('DecisionTreeRegressor:')
# print('The mean squared error for the DecisionTreeRegressor is:', mean_squared_error(y_test, y_pred))