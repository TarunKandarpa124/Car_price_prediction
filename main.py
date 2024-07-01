import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


df = pd.read_csv('train-data.csv')
print("shape: ", df.shape)
print("Null: ", df.isnull().sum())
print("df1:", df)

df.drop(["Unnamed: 0", 'New_Price'], axis=1, inplace=True)
print("df2:", df)

df.dropna(inplace=True)
print("df3:", df)
#df = df.drop_duplicates(inplace=True)
print("NULL1: ", df.isnull().sum())


df['Name'] = df['Name'].str.split(" ").str[0]
print(df['Name'])
df['Year'] = df['Year'].astype(str)
print(df['Year'].dtype)
print(df.dtypes)

print("==================================================================================================")

label_encoder = LabelEncoder()
encode_columns = df[["Name", "Location", "Year", "Fuel_Type", "Transmission", "Owner_Type"]]
for column in encode_columns:
    df[column] = label_encoder.fit_transform(df[column])
print("\nCOLUMNS COLUMNS COLUMNS \n", df[["Name", "Location", "Year", "Fuel_Type", "Transmission", "Owner_Type"]])

print("==================================================================================================")

df = df[df['Power'] != 'null bhp']



def replace_and_convert(N):
    for column in N:
        column = str(column)
        for i in df[column]:
            i = str(i)
            j = i.split(" ")[0]
            df[column] = df[column].replace(i, j)
        df[column] = df[column].astype(float)
        print(df[column])

N = ['Mileage', 'Engine', 'Power']

print(replace_and_convert(N))

print(df.dtypes)

#df.to_csv('output_file_1.csv', index=True)

print("================================================================================================================")

#df1 = pd.read_csv('output_file_1.csv')
#df1.dropna(inplace=True)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1
df_outliers_removed = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("Shape_new: ", df_outliers_removed.shape)


correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

'''
X = df1.drop('Price', axis=1)
y = df1['Price']

k = 'all'
selector = SelectKBest(score_func=f_classif, k=k)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support(indices=True)]
print("selected features: ", selected_features)

scores = selector.scores_
pvalues = selector.pvalues_
feature_scores = pd.DataFrame({'Features': X.columns, 'Scores': scores, 'P values': pvalues})
print(feature_scores)

selected_features_df1 = X[selected_features]
selected_features_df1['Price'] = y

print("Selected features of df1: ", selected_features_df1)
#selected_features_df1.to_csv('selected_features.csv', index=False)'''
correlation_matrix = df.corr()
print("Correlation: \n", correlation_matrix)

print("================================================================================================================================================")

scaler = StandardScaler()
#df = scaler.fit_transform(df)
#cols_name =['Name','Location','Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats','Price']
#df = pd.DataFrame(df, columns = cols_name)
#df.to_csv('scaled.csv', index=False)
#print(df)

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(df.drop('Price', axis=1), df['Price'], test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train_scaled)
X_test_scaled = scaler.fit_transform(X_test_scaled)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred_train = model.predict(X_train_scaled)
rmse_train = root_mean_squared_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

y_pred_test = model.predict(X_test_scaled)
rmse = root_mean_squared_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("\n========================Scaled Linear regression===============\n")
print("Mean: ", df['Price'].mean())
print('Root mean squared error of train: ', rmse_train)
print("Mean squared error of train: ", mse_train)
print('Coefficient of Determination of train: ', r2_train)
print('Root mean squared error of test: ', rmse)
print("Mean squared error of test: ", mse)
print('Coefficient of Determination of test: ', r2)


'''
X_train_noscale, X_test_noscale, y_train_noscale, y_test_noscale = train_test_split(selected_features_df1.drop('Price', axis=1), selected_features_df1['Price'], test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X_train_noscale, y_train_noscale)

y_pred_noscale = model1.predict(X_test_noscale)
rmse1 = root_mean_squared_error(y_test_noscale, y_pred_noscale)
mse1 = mean_squared_error(y_test_noscale, y_pred_noscale)
r2_1 = r2_score(y_test_noscale, y_pred_noscale)

print("\n========================Unscaled Linear regression===============\n")
print("Mean : ", df1['Price'].mean())
print("Root mean squared error : ", rmse1)
print("Mean squared error : ", mse1)
print("Coefficient of Determination : ", r2_1)'''

print("\n====================KNN Regression=================================\n")

knn = KNeighborsRegressor()
param_grid = {'n_neighbors': list(range(1, 31))}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_knn = grid_search.best_estimator_
best_k = grid_search.best_params_['n_neighbors']
best_score = -grid_search.best_score_
print("Best number of nearest neighbors: ", best_k)
print("Best cross validated MSE: ", best_score)

y_pred_knn = best_knn.predict(X_test_scaled)
rmse_knn = root_mean_squared_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
y_pred_knn_train = best_knn.predict(X_train_scaled)
rmse_knn_train = root_mean_squared_error(y_train, y_pred_train)
mse_knn_train = mean_squared_error(y_train, y_pred_train)
r2_knn_train = r2_score(y_train, y_pred_train)
print("Mean: ", df['Price'].mean())
print('Root Mean Squared Error knn test: ', rmse_knn)
print('Mean Squared Error knn test: ', mse_knn)
print('Coefficient of Determination (R^2) knn test: ', r2_knn)
print('Root Mean Squared Error knn train: ', rmse_knn_train)
print('Mean Squared Error knn train: ', mse_knn_train)
print('Coefficient of Determination (R^2) knn train: ', r2_knn_train)

'''''
testdata_df = pd.read_csv('test-data.csv')
testdata_df.drop(['Unnamed: 0', 'New_Price'], axis=1, inplace=True)
testdata_df.dropna(inplace=True)
print(testdata_df.head(20))
testdata_df['Year'] = testdata_df['Year'].astype(str)
testdata_df = testdata_df[testdata_df['Power'] != 'null bhp']
#test_data_df.to_csv('predicted_prices_of_the_cars.csv')
print(testdata_df.head(20))

N3 = ['Name', 'Mileage', 'Engine', 'Power']

for column in N3:
    testdata_df[column] = testdata_df[column].str.split(" ").str[0]

for column in N3:
    if column != 'Name':
        testdata_df[column] = testdata_df[column].str.replace(r'[^\d.]', '', regex=True)
        testdata_df[column] = testdata_df[column].astype(float)

for column in N3:
    if column in df.columns and df[column].dtype == 'int64':
        testdata_df[column] = LabelEncoder().fit_transform(testdata_df[column])

N4 = ['Name', 'Location', "Year", 'Fuel_Type', 'Transmission', 'Owner_Type']

def label_encoder_knn(N, df):
    for column in N:
        df[column] = LabelEncoder().fit_transform(df[column])

label_encoder_knn(N4, testdata_df)

print(testdata_df.head(25))

testdata_df_scale = scaler.transform(testdata_df)
predicted_price_knn = knn.predict(testdata_df_scale)
predicted_list_knn = []
for i in predicted_price_knn:
    predicted_list_knn.append(i)
print("Random Prediction: ", predicted_list_knn)
'''''
'''
print("\n==========================DECISION TREE===============================\n")

#dt_model = DecisionTreeRegressor(random_state=42)
#dt_model.fit(X_train_scaled, y_train)

print("\n========================PREDICTION==================================\n")

dummy_data = {
    'Name': ['Mercedes'],
    'Location': ['Chennai'],
    'Year': ['2014'],
    'Kilometers_Driven': [40000],
    'Fuel_Type': ['Petrol'],
    'Transmission': ['Manual'],
    'Owner_Type': ['First'],
    'Mileage': [18.0],
    'Engine': [1197],
    'Power': [83.0],
    'Seats': [5]
}

dummy_df = pd.DataFrame(dummy_data)

for column in dummy_df.columns:
    if column in df1.columns and df[column].dtype == 'int64':
        dummy_df[column] = LabelEncoder().fit_transform(dummy_df[column])


def label_encoder(N, df):
    for column in N:
        df[column] = LabelEncoder().fit_transform(df[column])

def one_hot_encoder(N, df):
    for column in N:
        df[column] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit_transform(df[column])

N = ['Name', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']

label_encoder(N, dummy_df)



dummy_scaled = scaler.transform(dummy_df[selected_features])

predicted_price = best_knn.predict(dummy_scaled)
print("Prediction knn: ", predicted_price[0])

predicted_price_dt = dt_model.predict(dummy_scaled)
print("Decision Tree Prediction: ", predicted_price_dt[0])

print("\n===========================SVR========================================\n")

svregressor = SVR()

svregressor.fit(X_train_scaled, y_train)
y_pred_svr = svregressor.predict(X_test_scaled)
rmse_svr = root_mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print("Mean : ", df['Price'].mean())
print("rmse svr : ", rmse_svr)
print("r2 svr : ", r2_svr)

dummy_data_1 = {
  'Name': ['Suzuki'],
  'Location': ['Madhurai'],
  'Year': ['2000'],
  'Kilometers_Driven': [1],
  'Fuel_Type': ['Diesel'],
  'Transmission': ['Automatic'],
  'Owner_Type': ['Third'],
  'Mileage': [1000000],
  'Engine': [1000000],
  'Power': [1000000],
  'Seats': [10]
}

dummy_df_1 = pd.DataFrame(dummy_data_1)

for column in dummy_df_1.columns:
    if column in df1.columns and df[column].dtype == 'int64':
        dummy_df_1[column] = LabelEncoder().fit_transform(dummy_df_1[column])


label_encoder(N, dummy_df_1)

dummy_scaled_svr = scaler.transform(dummy_df_1[selected_features])

predicted_price_svr = svregressor.predict(dummy_scaled_svr)
print("SVR Prediction: ", predicted_price_svr[0])'''

print("\n=============================Lasso regression==================================\n")

lasso = Lasso(alpha=0.1)

lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

rmse_lasso = root_mean_squared_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("Mean: ", df['Price'].mean())
print("Lasso rmse : ", rmse_lasso)
print("Lasso mse : ", mse_lasso)
print("Lasso r2_score : ", r2_lasso)

print("\n=============================Ridge Regression====================================\n")

ridge = Ridge(alpha=0.1)
ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)

rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("Mean: ", df['Price'].mean())
print("ridge rmse : ", rmse_ridge)
print("ridge r2_score : ", r2_ridge)

print("\n==========================Random Forest===================================\n")

random_forest = RandomForestRegressor(
    n_estimators=590,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

random_forest.fit(X_train_scaled, y_train)
y_pred_randomforest = random_forest.predict(X_test_scaled)
random_forest_rmse = root_mean_squared_error(y_test, y_pred_randomforest)
random_forest_r2 = r2_score(y_test, y_pred_randomforest)
print("Mean : ", df['Price'].mean())
print("Random forest RMSE : ", random_forest_rmse)
print("Random forest R2 : ", random_forest_r2)


test_data_df = pd.read_csv('test-data.csv')
test_data_df['Name_Car'] = test_data_df['Name']
test_data_df.to_csv('predicted_prices_new_price.csv')
test_data_df.drop(['Unnamed: 0', 'New_Price'], axis=1, inplace=True)
test_data_df.dropna(inplace=True)
print(test_data_df.head(20))
test_data_df['Year'] = test_data_df['Year'].astype(str)
test_data_df = test_data_df[test_data_df['Power'] != 'null bhp']
#test_data_df.to_csv('predicted_prices_of_the_cars.csv')
print(test_data_df.head(20))

N1 = ['Name', 'Mileage', 'Engine', 'Power']

for column in N1:
    test_data_df[column] = test_data_df[column].str.split(" ").str[0]

for column in N1:
    if column != 'Name':
        test_data_df[column] = test_data_df[column].str.replace(r'[^\d.]', '', regex=True)
        test_data_df[column] = test_data_df[column].astype(float)

for column in N1:
    if column in df.columns and df[column].dtype == 'int64':
        test_data_df[column] = Encoder().fit_transform(test_data_df[column])

def label_encoder(N, df):
    for column in N:
        df[column] = LabelEncoder().fit_transform(df[column])


N2 = ['Name', 'Location', "Year", 'Fuel_Type', 'Transmission', 'Owner_Type']
#test_data_df = one_hot_encoder(N, test_data_df)

label_encoder(N2, test_data_df)

print(test_data_df.head(25))

test_data_df_scale = scaler.transform(test_data_df.drop(columns=['Name_Car']))
predicted_price_random = random_forest.predict(test_data_df_scale)
predicted_list = []
for i in predicted_price_random:
    predicted_list.append(i)
print("Random Prediction: ", predicted_list)

test_data_df['Predicted_Price'] = predicted_price_random.round(2)
test_data_df.to_csv('Test-data-with-predicted-price.csv')
