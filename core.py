#Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Assumes dataset is saved as "mushrooms.csv" in the same directory (done by initial.py)
data = pd.read_csv("mushrooms.csv")
# Check for missing values
print("\n\n=================================================================================")
print("\nDataset loaded successfully.\n\n")
print("=================================================================================\n\n")
print("Checking for missing values...\n\n")
# Check for missing values
print("\nMissing values in dataset:")
print(data.isnull().sum())
if data.isnull().sum().any():
    print("There are missing values in the dataset. Please handle them before proceeding.")
    exit(1)
else:
    print("\n\nNo missing values found. Proceeding with the analysis...\n\n")
# and we need to convert them to numerical values for the model
# Initialize LabelEncoder
label_encoders = {}
# Loop through each column in the dataset
# and apply LabelEncoder to convert categorical values to numerical
# This is necessary for machine learning algorithms to work with categorical data
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Save encoders for possible inverse transform later
# Separate features and target
X = data.drop("class", axis=1)  # Features
y = data["class"]               # Target variable (0 = edible, 1 = poisonous)
# Split data into train and test sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Logistic Regression, Neural Network, SVM, Decision Tree, Random Forest
# Each model is initialized with default parameters
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}
# Train and evaluate each model
# Loop through each model in the models dictionary
# and train it on the training data
for name, model in models.items():
    print(f"\nTraining {name}...") # Print the name of the model being trained
    model.fit(X_train, y_train)                # Train the model
    y_pred = model.predict(X_test)             # Predict on test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(f"Accuracy of {name}: {accuracy:.4f}") # Print accuracy
    print("Classification Report:") # Print classification report
    print(classification_report(y_test, y_pred, target_names=["Edible", "Poisonous"]))
