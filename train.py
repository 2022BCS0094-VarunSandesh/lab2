import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (IMPORTANT: sep=';')
df = pd.read_csv("dataset/winequality-red.csv", sep=";")

# Print columns for debug (optional but safe)
print("Columns:", df.columns.tolist())

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save results
results = {
    "mse": mse,
    "r2_score": r2
}

with open("results.json", "w") as f:
    json.dump(results, f)

print("Training completed successfully")
