import pandas as pd
import joblib

# Load test data
test_df = pd.read_csv("C:/Users/saadk/Desktop/Week4 intern/test_Y3wMUE5_7gLdaTN.csv")

# Fill missing values just like training data
test_df['Gender'].fillna(test_df['Gender'].mode()[0], inplace=True)
test_df['Married'].fillna(test_df['Married'].mode()[0], inplace=True)
test_df['Dependents'].fillna(test_df['Dependents'].mode()[0], inplace=True)
test_df['Self_Employed'].fillna(test_df['Self_Employed'].mode()[0], inplace=True)
test_df['Credit_History'].fillna(test_df['Credit_History'].mode()[0], inplace=True)
test_df['Loan_Amount_Term'].fillna(test_df['Loan_Amount_Term'].mode()[0], inplace=True)
test_df['LoanAmount'].fillna(test_df['LoanAmount'].median(), inplace=True)

# Encode categorical variables (same as training)
from sklearn.preprocessing import LabelEncoder

cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
le = LabelEncoder()
for col in cat_cols:
    test_df[col] = le.fit_transform(test_df[col])

# Drop Loan_ID before prediction
X_test_final = test_df.drop(['Loan_ID'], axis=1)

# Load model
model = joblib.load("models/best_model.pkl")

# Predict
preds = model.predict(X_test_final)

# Convert predictions back to labels (1 = Y, 0 = N)
result = pd.DataFrame({
    'Loan_ID': test_df['Loan_ID'],
    'Loan_Status': ['Y' if pred == 1 else 'N' for pred in preds]
})

# Save predictions
result.to_csv("predictions.csv", index=False)
print(" Predictions saved to predictions.csv")
