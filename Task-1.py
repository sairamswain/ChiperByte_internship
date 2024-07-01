#Load and Combine the Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
Dataset = pd.read_csv("iris data.csv")

# Display the first 20 rows
print("First 20 rows of the dataset:")
print(Dataset.head(20))

# Display dataset summary
print("\nDataset Summary:")
print(Dataset.describe())

# Check for null values
print("\nNull values in the dataset:")
print(Dataset.isnull().sum())

# Data exploration
sns.pairplot(Dataset, hue='Species')
plt.suptitle('Pair Plot of Iris Dataset', y=1.02)
plt.show()

# Features and target variable
X = Dataset.drop(columns='Species')
y = Dataset['Species']


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Random Forest Classifier ###
# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf_model.predict(X_test)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy of Random Forest Classifier: {accuracy_rf:.2f}")

# Logistic Regression Classifier #
# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42, multi_class='ovr', max_iter=1000)  # Using One-vs-Rest strategy
lr_model.fit(X_train, y_train)

# Predict probabilities on test set
y_pred_prob_lr = lr_model.predict_proba(X_test)

# Calculate accuracy
accuracy_lr = accuracy_score(y_test, lr_model.predict(X_test))
print(f"Accuracy of Logistic Regression Classifier: {accuracy_lr:.2f}")

# Visualizations #
# Bar chart of predicted classes (Random Forest)
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
sns.countplot(y_pred_rf, palette='Set2')
plt.title('Predicted Class Distribution (Random Forest)')
plt.xlabel('Species')
plt.ylabel('Count')

# Pie chart of actual classes
plt.figure(figsize=(8, 6))
y_test.value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('Set2'), startangle=140)
plt.title('Actual Class Distribution')
plt.ylabel('')
plt.show()

# Logistic Regression: Plot predicted probabilities
plt.figure(figsize=(18, 6))
for i in range(len(lr_model.classes_)):
    sns.kdeplot(y_pred_prob_lr[:, i], label=lr_model.classes_[i])
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Logistic Regression: Predicted Probabilities')
plt.legend()
plt.show()

# Classification report (Logistic Regression)
print("\nClassification Report of Logistic Regression:")
print(classification_report(y_test, lr_model.predict(X_test)))
