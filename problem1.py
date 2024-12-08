# Importing necessary libraries
# kagglehub is used to download datasets; pandas is for data manipulation and exploration.
# Scikit-learn provides tools for model building, preprocessing, and evaluation.
# Matplotlib and seaborn are used for data visualization.
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Download the dataset
# Titanic dataset is downloaded from Kaggle using kagglehub.
# The dataset is suitable for classification problems as the target variable ("Survived") indicates
# whether a passenger survived the disaster, which is a binary classification problem.
path = kagglehub.dataset_download("zain280/titanic-data-set")
data_path = f"{path}/train.csv"
print("Dataset downloaded:", data_path)

# Step 2: Load the dataset
# The Titanic dataset is loaded into a pandas DataFrame for further exploration and preprocessing.
data = pd.read_csv(data_path)
print("First 5 rows of the dataset:\n", data.head())

# Step 3: Handle missing values
# Missing values are filled using forward fill ('ffill') based on the values in preceding rows.
# This ensures that no missing values remain in the dataset, which could affect model performance.
data.fillna(method='ffill', inplace=True)

# Step 4: Define target variable and features
# The "Survived" column is defined as the target variable to be predicted.
# Irrelevant or complex columns like "Name," "Ticket," and "Cabin" are dropped from the features.
target_column = "Survived"
X = data.drop(columns=[target_column, "Name", "Ticket", "Cabin"])  # Select features
y = data[target_column]  # Define the target variable

# Step 5: Encode categorical variables
# Categorical features are converted to numerical values using one-hot encoding.
# This ensures that machine learning models can process them effectively.
# The target variable ("Survived") is encoded as 0 and 1 using LabelEncoder.
X = pd.get_dummies(X, drop_first=True)  # Perform one-hot encoding on categorical variables
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode the target variable

# Step 6: Split the data into training and test sets
# The dataset is split into training (80%) and testing (20%) sets to evaluate model performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Scale the features
# Feature scaling is applied to normalize the feature values within a specific range.
# This step is especially critical for models like SVM which are sensitive to feature magnitudes.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit the scaler to training data and transform it
X_test = scaler.transform(X_test)  # Transform test data using the fitted scaler

# Step 8: Train a Decision Tree model
# A DecisionTreeClassifier is trained on the training data.
# This model is chosen for its interpretability and ability to handle categorical data.
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)  # Train the model on the training set

# Evaluate the Decision Tree model
# The trained Decision Tree model is evaluated using the test set.
y_pred_clf = clf.predict(X_test)  # Predict the target variable for the test set
print("\nDecision Tree Model Performance:")
print(classification_report(y_test, y_pred_clf))  # Output precision, recall, and F1 score

# Step 9: Train a Random Forest model
# A RandomForestClassifier is trained for comparison.
# Random Forest builds multiple decision trees and aggregates their results for better generalization.
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)  # Train the Random Forest model

# Evaluate the Random Forest model
# The trained Random Forest model is evaluated using the test set.
y_pred_rf = rf_clf.predict(X_test)  # Predict the target variable for the test set
print("\nRandom Forest Model Performance:")
print(classification_report(y_test, y_pred_rf))  # Output precision, recall, and F1 score

# Step 10: Train a Support Vector Machine (SVM) model
# An SVM classifier is trained for comparison.
# SVM is effective for high-dimensional spaces and suitable for binary classification tasks.
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train, y_train)  # Train the SVM model

# Evaluate the SVM model
# The trained SVM model is evaluated using the test set.
y_pred_svm = svm_clf.predict(X_test)  # Predict the target variable for the test set
print("\nSupport Vector Machine (SVM) Model Performance:")
print(classification_report(y_test, y_pred_svm))  # Output precision, recall, and F1 score

# Step 11: Compare model performance
# Accuracy and F1 score are compared across Decision Tree, Random Forest, and SVM.
# These metrics give insights into the classification performance of each model.
models = ['Decision Tree', 'Random Forest', 'SVM']
accuracy_scores = [accuracy_score(y_test, y_pred_clf), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_svm)]
f1_scores = [f1_score(y_test, y_pred_clf, average='weighted'), f1_score(y_test, y_pred_rf, average='weighted'), f1_score(y_test, y_pred_svm, average='weighted')]

# Accuracy comparison
plt.bar(models, accuracy_scores)
plt.title("Model Comparison - Accuracy")  # Title of the accuracy comparison chart
plt.ylabel("Accuracy")  # Label for the y-axis
plt.show()  # Display the chart

# F1 score comparison
plt.bar(models, f1_scores)
plt.title("Model Comparison - F1 Score")  # Title of the F1 score comparison chart
plt.ylabel("F1 Score")  # Label for the y-axis
plt.show()  # Display the chart

# Step 12: Visualize confusion matrices
# Confusion matrices are visualized for each model to assess classification performance.
# They show the number of true positive, true negative, false positive, and false negative predictions.
for model_name, model in zip(['Decision Tree', 'Random Forest', 'SVM'], [clf, rf_clf, svm_clf]):
    y_pred = model.predict(X_test)  # Predict the test set results for the current model
    cm = confusion_matrix(y_test, y_pred)  # Generate the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Plot the confusion matrix as a heatmap
    plt.title(f"Confusion Matrix - {model_name}")  # Set the plot title to the current model name
    plt.ylabel("True Label")  # Label the y-axis as "True Label"
    plt.xlabel("Predicted Label")  # Label the x-axis as "Predicted Label"
    plt.show()  # Display the plot