import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

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

gbr_hyp = {'max_depth': range(1,20)} 

gbr_hyp = GridSearchCV(GradientBoostingRegressor(), gbr_hyp, verbose=4)
gbr_hyp.fit(X_train, y_train)
print("GradientBoostingRegressor")
print("best score:", gbr_hyp.best_score_)
print("best parameters:", gbr_hyp.best_params_)

gbr = GradientBoostingRegressor(max_depth=10)
gbr.fit(X_train, y_train)

print("Training score", gbr.score(X_train, y_train))
print("Test score", gbr.score(X_test, y_test))

y_pred = gbr.predict(X_test)
print('GradientBoostingRegressor:')
print('The mean squared error for the GradientBoostingRegressor is:', mean_squared_error(y_test, y_pred))
