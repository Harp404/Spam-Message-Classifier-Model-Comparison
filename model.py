# Spam Message Classifier - Model Comparison and Training Script
# This script trains multiple machine learning models to classify spam vs ham messages
# and automatically selects the best performing model based on accuracy

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Welcome message and script description
print("=" * 80)
print("SPAM MESSAGE CLASSIFIER - MODEL COMPARISON & TRAINING")
print("=" * 80)
print(
    "This script trains and compares different ML models for spam email classification"
)
print("Models tested: Logistic Regression, Naive Bayes, XGBoost, Random Forest")
print("The best performing model will be automatically saved for future use\n")

# Get dataset path from user input
a = input(
    "Enter the path of the CSV file for data to train model! (Make sure it has columns 'Category','Message' (Default: data.csv): "
)
if a == "":
    a = os.path.join(os.getcwd(), "data.csv")

# Load and prepare the dataset
print(f"\nLoading dataset from: {a}")
data = pd.read_csv(a)
print(f"Dataset loaded successfully! Total samples: {len(data)}")

# Prepare target variable (y) - convert categories to binary labels
y_train = data["Category"]
y_train = y_train.map({"ham": 0, "spam": 1})  # ham=0 (not spam), spam=1 (spam)

# Prepare feature variable (x) - the message text
x_train = data["Message"]

# Text vectorization using TF-IDF
print("Converting text messages to numerical features using TF-IDF vectorization...")
tokenizer = TfidfVectorizer(max_features=5000)  # Use top 5000 features
x_train = tokenizer.fit_transform(x_train).toarray()
print(f"Text vectorization complete! Feature matrix shape: {x_train.shape}")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)
print(
    f"Data split complete - Training samples: {len(x_train)}, Testing samples: {len(x_test)}"
)

# Initialize different machine learning models
print("\nInitializing machine learning models...")
logistic_regression = LogisticRegression(penalty="l2", max_iter=500)
naive_bayes = MultinomialNB(alpha=0.5)
xgboost = xgb.XGBClassifier()
random_forest = RandomForestClassifier()

# Train all models
print("Training models... This may take a few moments.")
print("Training Logistic Regression...")
logistic_regression.fit(x_train, y_train)
print("Training Naive Bayes...")
naive_bayes.fit(x_train, y_train)
print("Training XGBoost...")
xgboost.fit(x_train, y_train)
print("Training Random Forest...")
random_forest.fit(x_train, y_train)
print("All models trained successfully!")

# Evaluate model performance on test set
print("\nEvaluating model performance...")
accuracy_l = accuracy_score(y_test, logistic_regression.predict(x_test)) * 100
accuracy_n = accuracy_score(y_test, naive_bayes.predict(x_test)) * 100
accuracy_x = accuracy_score(y_test, xgboost.predict(x_test)) * 100
accuracy_r = accuracy_score(y_test, random_forest.predict(x_test)) * 100

# Store accuracies in dictionary for easy comparison
accuracies = {
    "Logistic_Regression": accuracy_l,
    "Naive_Bayes": accuracy_n,
    "XGBoost": accuracy_x,
    "Random_Forest": accuracy_r,
}

# Display accuracy results
print("\n" + "=" * 50)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 50)
print(f"Logistic Regression: {accuracy_l:.2f}%")
print(f"Naive Bayes:         {accuracy_n:.2f}%")
print(f"XGBoost:             {accuracy_x:.2f}%")
print(f"Random Forest:       {accuracy_r:.2f}%")
print("=" * 50)

# Determine the best performing model
best_model_name = max(accuracies, key=accuracies.get)
model = {
    "Logistic_Regression": logistic_regression,
    "Naive_Bayes": naive_bayes,
    "XGBoost": xgboost,
    "Random_Forest": random_forest,
}[best_model_name]
print(
    f"🏆 BEST MODEL: {best_model_name} (Accuracy: {accuracies[best_model_name]:.2f}%)"
)

# Get model name from user for saving
name = input(f"\nEnter the name of the model (Default: model_{best_model_name}): ")
if name == "":
    name = "model"

# Save the best model and tokenizer to disk
print(f"\nSaving best model and tokenizer...")
with open(f"{name}_{best_model_name}.pkl", "wb") as f:
    pickle.dump(model, f)
    print(f"✅ Best model saved as: {name}_{best_model_name}.pkl")
with open(f"{name}_tokenizer_{best_model_name}.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer saved as: {name}_tokenizer_{best_model_name}.pkl")

# Interactive testing section
print("\n" + "=" * 50)
print("INTERACTIVE MESSAGE CLASSIFICATION")
print("=" * 50)
print("Test the trained model with your own messages!")
print("Enter messages to classify as spam or ham (type 'q' to quit)")
print("-" * 50)

while True:
    email = input("Enter the message to classify (q to quit): ")
    if email == "q":
        print("Thanks for using the Spam Message Classifier! 👋")
        break

    # Transform the input message using the same tokenizer
    email = tokenizer.transform([email]).toarray()

    # Make prediction using the best model
    h = model.predict(email)

    # Display result
    if h == 1:
        print("🚨 Classification: SPAM")
    else:
        print("✅ Classification: HAM (Not Spam)")
    print("-" * 30)
