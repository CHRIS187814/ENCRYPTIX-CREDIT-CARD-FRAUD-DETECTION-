# ENCRYPTIX-CREDIT-CARD-FRAUD-DETECTION-
# Credit Card Fraud Detection

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset contains anonymized transaction features and labels indicating fraudulent or legitimate transactions.

## Dataset
The dataset consists of:
- Features extracted from credit card transactions (e.g., amount, time, and anonymized variables)
- Labels: **0 (Legitimate Transaction), 1 (Fraudulent Transaction)**
- Highly imbalanced data (fraud cases are rare)

## Project Structure
```
├── data/                 # Raw and processed datasets
├── notebooks/            # Jupyter notebooks for exploration and model building
├── src/                  # Source code for training and evaluation
├── models/               # Trained models and evaluation results
├── reports/              # Generated reports and insights
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
├── main.py               # Main script for training and prediction
```

## Installation
To set up the project, clone the repository and install dependencies:
```sh
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

## Usage
Run the main script to train and evaluate the model:
```sh
python main.py
```

## Machine Learning Model
The project explores different classification models, including:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Neural Networks

## Techniques Used
To handle class imbalance and improve accuracy, we apply:
- **Oversampling (SMOTE)**
- **Undersampling**
- **Anomaly Detection Methods**
- **Feature Engineering and Selection**

## Results
The models are evaluated using:
- **Accuracy, Precision, Recall, and F1-score**
- **Confusion Matrix**
- **ROC-AUC Curve**

## Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For any questions, feel free to reach out via GitHub or email: **chrisblessan72@gmail.com**

