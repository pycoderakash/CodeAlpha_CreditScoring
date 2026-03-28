# CodeAlpha_CreditScoring
# ==============================
# CREDIT SCORING MODEL PROJECT
# ==============================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ------------------------------
# Step 2: Data Cleaning
# ------------------------------
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# ------------------------------
# Step 3: Encoding
# ------------------------------
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# ------------------------------
# Step 4: Basic Info
# ------------------------------
print("Dataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

# ------------------------------
# Step 5: Visualization
# ------------------------------
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("0 = Not Survived, 1 = Survived")
plt.ylabel("Count")
plt.show()

# ------------------------------
# Step 6: Features & Target
# ------------------------------
X = df.drop("Survived", axis=1)
y = df["Survived"]

# ------------------------------
# Step 7: Train-Test Split (FIXED)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Step 8: Model Training (FIXED)
# ------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Step 9: Prediction & Evaluation
# ------------------------------
y_pred = model.predict(X_test)

print("\nModel Performance:\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ------------------------------
# Step 10: Feature Importance
# ------------------------------
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

importance_df.plot(kind="bar", x="Feature", y="Importance")
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()
