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

X = np.asarray(X).astype(np.float32)
y = np.asarray(y).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state=19)

y_train = np.reshape(y_train, (y_train.shape[0],1))
y_test = np.reshape(y_test, (y_test.shape[0],1))

print('Shape of training data:    ', X_train.shape)
print('Shape of training labels:  ', y_train.shape)
print('Shape of test data:        ', X_test.shape)
print('Shape of test labels:      ', y_test.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, input_shape=(52, )),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='relu'),
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

model.fit(X_train, y_train, epochs=1000)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
