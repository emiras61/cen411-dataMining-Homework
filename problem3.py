# Importing necessary libraries
# KaggleHub is used to download datasets; pandas is used for data manipulation and analysis.
# Scikit-learn provides tools for splitting data, preprocessing, model training, and evaluation.
# Matplotlib is used for visualizing the results.
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Download the dataset
# The Diabetes dataset is ideal for regression problems as it contains medical indicators
# that can predict the outcome variable (e.g., diabetes presence or severity).
path = kagglehub.dataset_download("vikasukani/diabetes-data-set")
data_path = f"{path}/diabetes-dataset.csv"
print("Dataset downloaded:", data_path)

# Step 2: Load the dataset
# The dataset is loaded into a pandas DataFrame to facilitate preprocessing and exploration.
data = pd.read_csv(data_path)
print("First 5 rows of the dataset:\n", data.head())

# Step 3: Check for missing values and handle them
# Missing values can negatively impact the model's learning process. If any are found,
# they are filled using the forward fill method to maintain data continuity.
if data.isnull().sum().any():
    print("Missing values detected. Filling missing values...")
    data.fillna(method='ffill', inplace=True)

# Step 4: Define the target variable and features
# "Outcome" is chosen as the target variable because it represents the outcome we want to predict.
# All other columns are used as input features.
target_column = "Outcome"
X = data.drop(columns=[target_column])  # Features (independent variables)
y = data[target_column]  # Target variable (dependent variable)

# Step 5: Split the data into training and test sets
# The dataset is divided into 80% training and 20% testing to ensure the model is evaluated
# on unseen data, providing an unbiased measure of its performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Scale the features
# Feature scaling ensures that all features contribute equally to the model.
# StandardScaler standardizes the data by subtracting the mean and dividing by the standard deviation.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fitting and scaling the training data
X_test = scaler.transform(X_test)  # Scaling the test data using the same scaler

# Step 7: Train a Linear Regression model
# Linear regression is a fundamental algorithm for regression tasks, which assumes a linear
# relationship between the features and the target variable.
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)  # Training the linear regression model

# Make predictions using the trained Linear Regression model
y_pred_linear = linear_reg.predict(X_test)

# Evaluate the performance of the Linear Regression model
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print("\nLinear Regression Performance:")
print(f"Mean Squared Error (MSE): {mse_linear:.2f}")
print(f"Mean Absolute Error (MAE): {mae_linear:.2f}")
print(f"R^2 Score: {r2_linear:.2f}")

# Step 8: Train a Ridge Regression model
# Ridge regression is an extension of linear regression that includes L2 regularization,
# which penalizes large coefficients to reduce overfitting.
ridge_reg = Ridge(alpha=1.0)  # The alpha parameter controls the strength of regularization.
ridge_reg.fit(X_train, y_train)  # Training the Ridge regression model

# Make predictions using the trained Ridge Regression model
y_pred_ridge = ridge_reg.predict(X_test)

# Evaluate the performance of the Ridge Regression model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print("\nRidge Regression Performance:")
print(f"Mean Squared Error (MSE): {mse_ridge:.2f}")
print(f"Mean Absolute Error (MAE): {mae_ridge:.2f}")
print(f"R^2 Score: {r2_ridge:.2f}")

# Step 9: Train a Random Forest Regressor
# Random Forest is an ensemble learning method that uses multiple decision trees
# to capture non-linear relationships in the data and improve predictive accuracy.
rf_reg = RandomForestRegressor(random_state=42, n_estimators=100)  # Using 100 trees
rf_reg.fit(X_train, y_train)  # Training the Random Forest regressor

# Make predictions using the trained Random Forest Regressor
y_pred_rf = rf_reg.predict(X_test)

# Evaluate the performance of the Random Forest Regressor
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("\nRandom Forest Regression Performance:")
print(f"Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"Mean Absolute Error (MAE): {mae_rf:.2f}")
print(f"R^2 Score: {r2_rf:.2f}")

# Step 10: Compare the performance of different models
# Performance metrics such as Mean Squared Error (MSE) and R^2 Score are compared across
# Linear Regression, Ridge Regression, and Random Forest models.
models = ['Linear Regression', 'Ridge Regression', 'Random Forest']
mse_scores = [mse_linear, mse_ridge, mse_rf]
r2_scores = [r2_linear, r2_ridge, r2_rf]

# Visualizing MSE comparison
plt.bar(models, mse_scores)
plt.title("Model Comparison - Mean Squared Error")
plt.ylabel("MSE")
plt.show()

# Visualizing R^2 comparison
plt.bar(models, r2_scores)
plt.title("Model Comparison - R^2 Score")
plt.ylabel("R^2")
plt.show()