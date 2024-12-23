Feature Analysis and Price Prediction for Handsets
Project Overview
This project focuses on predicting handset prices based on key features such as camera specifications, memory, battery capacity, and more. Using machine learning techniques, we analyze feature importance and train various models to achieve accurate price predictions. The project highlights feature engineering, model evaluation, and hyperparameter optimization to identify the best-performing model.

Features
Data Preprocessing:

Cleaning and transforming raw data for model readiness.
Handling missing values and scaling numeric features.
Encoding categorical variables.
Feature Selection:

Recursive Feature Elimination (RFE).
Mutual Information Regression.
Lasso Regression for feature importance.
Model Training and Evaluation:

Models implemented:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
Performance metrics: MAE, RMSE, and R² Score.
Hyperparameter Tuning:

Used GridSearchCV to optimize the Random Forest Regressor.
Results:

Tuned Random Forest Regressor identified as the best model.
Visualization of feature importance and predicted vs. actual prices.
Technologies Used
Python
Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, joblib
Future Enhancements
Include additional features like brand, market demand, and user ratings.
Experiment with advanced models such as XGBoost and deep learning.
Develop a user-friendly interface for real-time predictions.
