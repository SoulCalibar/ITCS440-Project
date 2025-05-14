import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline

# Load the dataset
print("Loading and preprocessing data...")
data = pd.read_csv("mushrooms.csv")

# Preprocessing: Encode all categorical features
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Save encoders for possible inverse transform later

# if the data.columns[0] is there not null
if data.columns[0] is None:
    raise ValueError("The first column of the dataset is None. Please check the dataset.")

# Assuming the first column is the target variable
X = data.drop(data.columns[0], axis=1)
y = data[data.columns[0]]  # Assuming the first column is the target variable

# Split data into train and test sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 1. Optimized Logistic Regression
print("\nTraining Optimized Logistic Regression...")
log_reg_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000, 2000],
    'class_weight': [None, 'balanced']
}

log_reg = GridSearchCV(LogisticRegression(), log_reg_params, cv=5, scoring='accuracy')
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Best parameters for Logistic Regression: {log_reg.best_params_}")
print(f"Accuracy of Optimized Logistic Regression: {accuracy_log_reg:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_log_reg, target_names=["Edible", "Poisonous"]))

# 2. Optimized SVM
print("\nTraining Optimized SVM...")
svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

svm = GridSearchCV(SVC(), svm_params, cv=5, scoring='accuracy')
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Best parameters for SVM: {svm.best_params_}")
print(f"Accuracy of Optimized SVM: {accuracy_svm:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=["Edible", "Poisonous"]))

# 3. Optimized Neural Network
print("\nTraining Optimized Neural Network...")
nn_params = {
    'hidden_layer_sizes': [(50, 30), (100, 50), (100, 50, 25)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [1000]
}

nn = GridSearchCV(MLPClassifier(), nn_params, cv=3, scoring='accuracy')
nn.fit(X_train_scaled, y_train)
y_pred_nn = nn.predict(X_test_scaled)
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print(f"Best parameters for Neural Network: {nn.best_params_}")
print(f"Accuracy of Optimized Neural Network: {accuracy_nn:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_nn, target_names=["Edible", "Poisonous"]))

# 4. Gradient Boosting Classifier (new model)
print("\nTraining Gradient Boosting Classifier...")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Accuracy of Gradient Boosting: {accuracy_gb:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_gb, target_names=["Edible", "Poisonous"]))

# 5. Ensemble Voting Classifier (combines multiple models)
print("\nTraining Ensemble Voting Classifier...")
estimators = [
    ('log_reg', LogisticRegression(max_iter=1000)),
    ('svm', SVC(probability=True)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100))
]
voting_clf = VotingClassifier(estimators=estimators, voting='soft')
voting_clf.fit(X_train_scaled, y_train)
y_pred_voting = voting_clf.predict(X_test_scaled)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print(f"Accuracy of Voting Classifier: {accuracy_voting:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_voting, target_names=["Edible", "Poisonous"]))

# Print summary of all models
print("\n=== Model Accuracy Summary ===")
models = {
    "Optimized Logistic Regression": accuracy_log_reg,
    "Optimized SVM": accuracy_svm,
    "Optimized Neural Network": accuracy_nn,
    "Gradient Boosting": accuracy_gb,
    "Ensemble Voting Classifier": accuracy_voting
}

for model, accuracy in models.items():
    print(f"{model}: {accuracy:.4f}")

# Feature importance from Gradient Boosting (often provides better feature importance than Random Forest)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 most important features (Gradient Boosting):")
print(feature_importances.head(10))
