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
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# print an error if the file isn't there
if not os.path.exists("mushrooms.csv"):
    print("Error: mushrooms.csv not found in the current directory.")
    exit(1)


# save the model to a file
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

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

print("\n\n=================================================================================")
# print the column names and their corresponding numerical values
print("Column names")
print(data.columns)
print("\n\n")
# ask the user what the target variable is
target_variable = input("Enter the name of the target variable (last column): ")


# Check if the target variable is in the dataset
if target_variable not in data.columns:
    # raise ValueError(f"Target variable '{target_variable}' not found in the dataset.")
    # set target variable to the last column
    target_variable = data.columns[-1]

# Separate features and target
X = data.drop(target_variable, axis=1)  # Features            # Target variable (0 = edible, 1 = poisonous)
y = data[target_variable]

# Split dataset into training (66%) and testing (34%) sets
# random_state=42 ensures reproducibility of the random split across multiple runs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)

# Initialize OneHotEncoder for categorical variable transformation
# sparse_output=False returns a dense numpy array instead of sparse matrix
# handle_unknown='ignore' safely handles any new categories in test data
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Convert categorical variables to binary format (one-hot encoding)
# This is necessary for machine learning algorithms to work with categorical data
# The fit_transform method learns the categories from the training data
# and transforms the data into a binary format
# For example, if the column 'color' has values ['red', 'blue', 'green'],
# 'color': ['red'] -> [1, 0, 0] (red, blue, green)
X_train_encoded = encoder.fit_transform(X_train)

# Apply the same encoding to test data using the categories learned from training
# This ensures consistency between training and test transformations
X_test_encoded = encoder.transform(X_test)



# Feature scaling; for better performance

# StandardScaler standardizes features by removing the mean and scaling to unit variance
# This is important for algorithms like SVM and Neural Networks that are sensitive to the scale of the features
# It transforms the data to have a mean of 0 and a standard deviation of 1
# For example, if the feature 'height' has values [1.5, 2.0, 2.5],
# StandardScaler will transform it to have a mean of 0 and std of 1
# The fit_transform method learns the mean and std from the training data and transforms the data
# The transform method applies the same transformation to the test data
# This is important to avoid data leakage and ensure that the model generalizes well
scaler = StandardScaler()
print(X_train_encoded)
print(X_test_encoded)
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Logistic Regression, Neural Network, SVM, Decision Tree, Random Forest
#logistic vs linear regression: Logistic regression is a classification algorithm, while linear regression is used for continous variables.
# Each model is initialized with default parameters
models = {
    #1000 iterations for convergence, prevents warnings but increases training time
    "Logistic Regression": LogisticRegression(max_iter=1000),
    # MLPClassifier is a multi-layer perceptron classifier
    # 50 neurons in the first hidden layer and 30 in the second
    # max_iter=1000 ensures the model trains for a maximum of 1000 iterations
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    # n_estimators=100 means it will use 100  trees, balanced for time and accuracy and performance
    "Random Forest": RandomForestClassifier(n_estimators=100)
}
# Train and evaluate each model
# Loop through each model in the models dictionary
# and train it on the training data

for name, model in models.items():
    print(f"\nTraining {name}...") # Print the name of the model being trained
    model.fit(X_train_scaled, y_train)                # Train the model
    y_pred = model.predict(X_test_scaled)             # Predict on test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(f"Accuracy of {name}: {accuracy:.4f}") # Print accuracy
    print("Classification Report:") # Print classification report

    # Get the unique classes from your target variable
    unique_classes = sorted(label_encoders[target_variable].classes_)
    print(classification_report(y_test, y_pred, target_names=unique_classes))
print("\n=================================================================================")

save_model(models["Logistic Regression"], "logistic_regression.joblib")
save_model(models["Neural Network"], "neural_network.joblib")
save_model(models["Support Vector Machine"], "support_vector_machine.joblib")
save_model(models["Decision Tree"], "decision_tree.joblib")
save_model(models["Random Forest"], "random_forest.joblib")
