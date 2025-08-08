from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import os

os.makedirs("models", exist_ok=True)  # Create the folder if it doesn't exist


# Load preprocessed data
import pandas as pd
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train.values.ravel())
    preds = model.predict(X_train)
    
    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_train, preds))
    print("Precision:", precision_score(y_train, preds))
    print("Recall:", recall_score(y_train, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_train, preds))

# Save the best model manually (e.g., Random Forest if best)
best_model = models["Random Forest"]
joblib.dump(best_model, "models/best_model.pkl")
print("\n Best model saved to 'models/best_model.pkl'")