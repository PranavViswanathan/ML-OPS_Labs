import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# ---------- Load Titanic Dataset from seaborn ----------
df = sns.load_dataset("titanic")

# Select relevant features
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
df = df[features + ["survived"]]

# Map categorical to numeric
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["embarked"] = df["embarked"].map({"C": 0, "Q": 1, "S": 2})

# Drop rows with missing target
df = df.dropna(subset=["survived"])

# Fill missing values with median
df = df.fillna(df.median(numeric_only=True))

X = df.drop("survived", axis=1)
y = df["survived"]

# ---------- Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Train Model ----------
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# ---------- Evaluate ----------
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Validation Accuracy: {acc:.4f}")

# ---------- Save ----------
joblib.dump(model, "titanic_gb.pkl")
print("Model saved to titanic_gb.pkl")
