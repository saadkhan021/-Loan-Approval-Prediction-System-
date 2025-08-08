# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\saadk\Desktop\Week4 intern\train_u6lujuX_CVtuZ9i.csv")

# Step 1: Basic info
print("Shape:", df.shape)
print(df.info())
print(df.isnull().sum())

# Step 2: Fill missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Step 3: Encode categorical columns
cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 
            'Property_Area', 'Loan_Status', 'Dependents']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Step 4: Define features and target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Save preprocessed data (for reference)
X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)

print("Preprocessing complete.")
