import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define file paths
zip_path = "C:/Users/bless/downloads/creditcard.csv.zip"
unzip_dir = "C:/Users/bless/downloads"

# Extract the CSV file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_dir)

# Load dataset
df = pd.read_csv(os.path.join(unzip_dir, "creditcard.csv"))

# Exploratory Data Analysis
print(df.head())
print(df.info())
print(df.describe())
print("Class Distribution:\n", df['Class'].value_counts())

# Splitting features and labels
X = df.drop(columns=['Class'])  # Features
y = df['Class']  # Target variable

# Normalize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualizing Class Distribution Before and After SMOTE
plt.figure(figsize=(10, 5))
sns.countplot(x=y, label="Original", color='blue', alpha=0.5)
sns.countplot(x=y_resampled, label="After SMOTE", color='red', alpha=0.5)
plt.legend()
plt.title("Class Distribution Before and After SMOTE")
plt.show()
