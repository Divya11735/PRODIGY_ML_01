import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "house_data.csv")

data = pd.read_csv(DATA_PATH)

# Select features and target
X = data[['rm', 'ptratio', 'lstat']]
y = data['medv']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save predictions
predictions = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

predictions.to_csv("../outputs/predictions.csv", index=False)
print("Predictions saved in outputs/predictions.csv")

import joblib
import os

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "house_price_model.pkl")
joblib.dump(model, MODEL_PATH)

print("Model saved at:", MODEL_PATH)
