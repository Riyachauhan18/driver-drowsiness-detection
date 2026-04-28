import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load dataset
df = pd.read_csv("drowsiness_dataset.csv")

X = df[["EAR", "MAR"]]
y = df["Label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier()
}

best_model = None
best_score = 0

print("\n--- Model Results ---\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{name}:")
    print(f"Accuracy = {acc:.4f}")
    print(f"F1 Score = {f1:.4f}")
    print("----------------------")

    if f1 > best_score:
        best_score = f1
        best_model = model

# Save best model
joblib.dump(best_model, "drowsiness_model.pkl")

print("\n✅ Best model saved as drowsiness_model.pkl")