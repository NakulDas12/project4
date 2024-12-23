import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import joblib


file_path = '/Users/nakuldas/Documents/pythonProject4/Processed_Flipdata - Processed_Flipdata.csv'
dataset = pd.read_csv(file_path)


print("First few rows of the dataset:")
print(dataset.head())


print("\nDataset Information:")
dataset.info()


print("\nMissing Values in Each Column:")
print(dataset.isnull().sum())


print("\nSummary Statistics:")
print(dataset.describe())


sns.pairplot(dataset)
plt.title('Pairplot of Features')
plt.show()


dataset['Rear Camera'] = dataset['Rear Camera'].str.replace('MP', '').astype(int)
dataset['Front Camera'] = dataset['Front Camera'].str.replace('MP', '').astype(int)


dataset['Prize'] = dataset['Prize'].str.replace(',', '').astype(float)


print("\nData Types After Conversion:")
print(dataset.dtypes)


dataset.fillna(dataset.mean(numeric_only=True), inplace=True)


dataset = pd.get_dummies(dataset, columns=['Colour', 'Processor_'], drop_first=True)


scaler = MinMaxScaler()
numerical_cols = ['Memory', 'RAM', 'Battery_', 'Mobile Height', 'Rear Camera', 'Front Camera']
dataset[numerical_cols] = scaler.fit_transform(dataset[numerical_cols])


print("\nFinal Preprocessed Dataset:")
print(dataset.head())


X = dataset.drop(['Prize', 'Model'], axis=1, errors='ignore')  # Drop target and any non-numeric columns
y = dataset['Prize']


model = LinearRegression()
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X, y)
print("\nSelected Features by RFE:")
selected_features_rfe = X.columns[fit.support_]
print(selected_features_rfe)


mi_scores = mutual_info_regression(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)
mi_scores.sort_values(ascending=False, inplace=True)


plt.figure(figsize=(12, 6))
mi_scores.plot(kind='bar')
plt.title('Mutual Information Scores')
plt.ylabel('Score')
plt.show()


lasso = LassoCV(cv=5)
lasso.fit(X, y)
lasso_coef = pd.Series(lasso.coef_, index=X.columns)
lasso_coef = lasso_coef[lasso_coef != 0]  # Keep only non-zero coefficients
print("\nFeatures Selected by Lasso Regression:")
print(lasso_coef)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model_performance = {}


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}


for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Corrected line
    r2 = r2_score(y_test, y_pred)


    model_performance[model_name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}


performance_df = pd.DataFrame(model_performance).T
print("\nModel Performance Metrics:")
print(performance_df)


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
best_model_name = 'Random Forest (Tuned)'


y_pred_best = best_model.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))

r2_best = r2_score(y_test, y_pred_best)

print(f"\nBest Model: {best_model_name}")
print(f"MAE: {mae_best}, RMSE: {rmse_best}, R²: {r2_best}")


importances = best_model.feature_importances_
feature_names = X.columns


feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Random Forest Model')
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Reference line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices')
plt.show()


joblib.dump(best_model, 'best_random_forest_model.pkl')
print("Best model saved as 'best_random_forest_model.pkl'.")