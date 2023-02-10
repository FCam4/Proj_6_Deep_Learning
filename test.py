import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Feature Scalling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

test = pd.read_csv('C:\\Users\\sebas\Downloads\\archive\\Google_Playstore.csv')

# test.to_csv('Google_Playstore.pkl', index = False, encoding='utf-8')

# df_1 = test.iloc[:231000,:]
# df_2 = test.iloc[231000:462000,:]
# df_3 = test.iloc[462000:693000,:]
# df_4 = test.iloc[693000:924000,:]
# df_5 = test.iloc[924000:1155000,:]
# df_6 = test.iloc[1155000:1386000,:]
# df_7 = test.iloc[1386000:1617000,:]
# df_8 = test.iloc[1617000:1848000,:]
# df_9 = test.iloc[1848000:2079000,:]
# df_10 = test.iloc[2079000:,:]

# df_1.to_csv('Google_Playstore1.csv', index = False)
# df_2.to_csv('Google_Playstore2.csv', index = False)
# df_3.to_csv('Google_Playstore3.csv', index = False)
# df_4.to_csv('Google_Playstore4.csv', index = False)
# df_5.to_csv('Google_Playstore5.csv', index = False)
# df_6.to_csv('Google_Playstore6.csv', index = False)
# df_7.to_csv('Google_Playstore7.csv', index = False)
# df_8.to_csv('Google_Playstore8.csv', index = False)
# df_9.to_csv('Google_Playstore9.csv', index = False)
# df_10.to_csv('Google_Playstore10.csv', index = False)

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

# # Train the scaler, which standarizes all the features to have mean=0 and unit variance
# sc = StandardScaler()
# sc.fit(X_train)

# # Apply the scaler to the X training data
# X_train_std = sc.transform(X_train)

# # Apply the SAME scaler to the X test data
# X_test_std = sc.transform(X_test)

# # Min Max Scaler
# scaler_min_max = MinMaxScaler()
# scaler_min_max.fit(X_train)

# # Apply the scaler to the X training data
# X_train_std_min_max = scaler_min_max.transform(X_train)

# # Apply the SAME scaler to the X test data
# X_test_std_min_max = scaler_min_max.transform(X_test)

# KNeighborRegressor
# Neigh = KNeighborsRegressor().fit(X_train, y_train)
# print('KNN REGRESSOR:')
# print('The train score for the KNeighborRegressor is:', Neigh.score(X_train, y_train))
# print('The test score for the KNeighborRegressor is:', Neigh.score(X_test, y_test))
# print('\n')

# # Linear Regression
# LR = LinearRegression().fit(X_train, y_train)
# print('LINEAR REGRESSION:')
# print('The train score for the Linear Regression is:', LR.score(X_train, y_train))
# print('The test score for the Linear Regression is:', LR.score(X_test, y_test))
# print('\n')


# KNeighborRegressor
# Neigh = KNeighborsRegressor().fit(X_train_std, y_train)
# print('Standard Scaler:')
# print('KNN REGRESSOR:')
# # print('The train score for the KNeighborRegressor is:', Neigh.score(X_train_std, y_train))
# print('The test score for the KNeighborRegressor is:', Neigh.score(X_test_std, y_test))
# print('\n')

# # Linear Regression
# LR = LinearRegression().fit(X_train_std, y_train)
# print('LINEAR REGRESSION:')
# print('The train score for the Linear Regression is:', LR.score(X_train_std, y_train))
# print('The test score for the Linear Regression is:', LR.score(X_test_std, y_test))
# print('\n')

# KNeighborRegressor
# Neigh = KNeighborsRegressor().fit(X_train_std_min_max, y_train)
# print('Min Max\n')
# print('KNN REGRESSOR:')
# print('The train score for the KNeighborRegressor is:', Neigh.score(X_train_std_min_max, y_train))
# print('The test score for the KNeighborRegressor is:', Neigh.score(X_test_std_min_max, y_test))
# print('\n')

# # Linear Regression
# LR = LinearRegression().fit(X_train_std_min_max, y_train)
# print('LINEAR REGRESSION:')
# print('The train score for the Linear Regression is:', LR.score(X_train_std_min_max, y_train))
# print('The test score for the Linear Regression is:', LR.score(X_test_std_min_max, y_test))
# print('\n')

knn_hyp = {'n_neighbors': range(2,20), 'weights': ['uniform', 'distance']}

dtc_hyp = {'max_depth': range(1,20)} 
    

gs_knn = GridSearchCV(KNeighborsRegressor(), knn_hyp)
gs_knn.fit(X_train, y_train)
print("KNN")
print("best score:", gs_knn.best_score_)
print("best parameters:",gs_knn.best_params_)

gs_dtr = GridSearchCV(DecisionTreeRegressor(), dtc_hyp)
gs_dtr.fit(X_train, y_train)
print("\nDecision Tree Regressor")
print("best score:", gs_dtr.best_score_)
print("best parameters:",gs_dtr.best_params_)





