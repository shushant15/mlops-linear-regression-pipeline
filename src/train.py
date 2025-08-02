import joblib
import os
import sys
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_data
warnings.filterwarnings('ignore', category=RuntimeWarning)

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e10, neginf=-1e10)
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=1e10, neginf=-1e10)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error (Loss): {mse:.4f}")
    
    joblib.dump({'model': model, 'scaler': scaler}, 'regression_model.joblib')
    
    return model, r2, mse

if __name__ == "__main__":
    train_model()