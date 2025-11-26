import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
print("Loading dataset...")
df = pd.read_csv('train_final.csv')

# Prepare features and target
X = df.drop('price_in_euro', axis=1)
y = df['price_in_euro']

print(f"Dataset shape: {df.shape}")
print(f"Features: {X.shape[1]}")
print(f"Target range: €{y.min():,.2f} - €{y.max():,.2f}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training R² score: {train_score:.4f}")
print(f"Testing R² score: {test_score:.4f}")

# Save model
joblib.dump(model, 'car_price_model.pkl')
print("Model saved as 'car_price_model.pkl'")

# Save feature names
import json
with open('feature_names.json', 'w') as f:
    json.dump(X.columns.tolist(), f)
print("Feature names saved as 'feature_names.json'")