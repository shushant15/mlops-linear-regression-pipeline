import joblib
import numpy as np
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_data

warnings.filterwarnings('ignore', category=RuntimeWarning)

def predict():
    """Loads the trained model and makes predictions on the test set."""
    try:
        loaded_artifacts = joblib.load("regression_model.joblib")
        model = loaded_artifacts["model"]
        scaler = loaded_artifacts["scaler"]
    except FileNotFoundError:
        print("Error: Trained model or scaler not found. Please run train.py first.")
        return

    _, X_test, _, _ = load_data()
    
    # Handle potential NaN/Inf values in test data
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)

    # Scale the test data using the loaded scaler
    X_test_scaled = scaler.transform(X_test)

    predictions = model.predict(X_test_scaled)

    print("\nSample predictions from the trained model:")
    for i, pred in enumerate(predictions[:5]):
        print(f"Prediction {i+1}: {pred:.2f}")

if __name__ == "__main__":
    predict()