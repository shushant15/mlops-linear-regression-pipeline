import joblib
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils import load_data


def quantize_parameters(model):
    coef = model.coef_
    intercept = model.intercept_

    joblib.dump({
        'coef': coef,
        'intercept': intercept
    }, 'unquant_params.joblib')

    min_coef = np.min(coef)
    max_coef = np.max(coef)
    scale_coef = (max_coef - min_coef) / 255.0
    zero_point_coef = 0 - (min_coef / scale_coef)

    if np.isscalar(intercept):
        min_intercept = intercept
        max_intercept = intercept
        if intercept < 0:
            min_intercept = intercept
            max_intercept = 0
        else:
            min_intercept = 0
            max_intercept = intercept
    else:
        min_intercept = np.min(intercept)
        max_intercept = np.max(intercept)

    scale_intercept = (max_intercept - min_intercept) / 255.0 if (max_intercept - min_intercept) != 0 else 1.0
    zero_point_intercept = 0 - (min_intercept / scale_intercept) if scale_intercept != 0 else 0

    quantized_coef = np.round((coef / scale_coef) + zero_point_coef).astype(np.uint8)
    quantized_intercept = np.round((intercept / scale_intercept) + zero_point_intercept).astype(np.uint8)
    
    joblib.dump({
        'quantized_coef': quantized_coef,
        'quantized_intercept': quantized_intercept,
        'scale_coef': scale_coef,
        'zero_point_coef': zero_point_coef,
        'scale_intercept': scale_intercept,
        'zero_point_intercept': zero_point_intercept,
    }, 'quant_params.joblib', compress=3)

    return quantized_coef, quantized_intercept, scale_coef, zero_point_coef, scale_intercept, zero_point_intercept

def dequantize_and_infer(quantized_coef, quantized_intercept, scale_coef, zero_point_coef, scale_intercept, zero_point_intercept, scaler):

    dequantized_coef = (quantized_coef - zero_point_coef) * scale_coef
    dequantized_intercept = (quantized_intercept - zero_point_intercept) * scale_intercept

    class QuantizedLinearRegression:
        def __init__(self, coef, intercept):
            self.coef_ = coef
            self.intercept_ = intercept

        def predict(self, X):
            return np.dot(X, self.coef_) + self.intercept_

    dequantized_model = QuantizedLinearRegression(dequantized_coef, dequantized_intercept)

    _, X_test, _, y_test = load_data()
    X_test_scaled = scaler.transform(X_test)

    predictions = dequantized_model.predict(X_test_scaled)
    print("\nSample predictions with de-quantized weights:")
    print(predictions[:5])
    
     # Verification of quantized model
    r2_dequantized = r2_score(y_test, predictions)
    print(f"R2 Score (de-quantized model): {r2_dequantized:.4f}")
    
    return predictions

if __name__ == "__main__":
    try:
        model_artifacts = joblib.load('regression_model.joblib')
        model = model_artifacts['model']
        scaler = model_artifacts['scaler']
    except FileNotFoundError:
        print("Error: 'regression_model.joblib' not found. Please run train.py first.")
        exit()

    q_coef, q_intercept, s_coef, zp_coef, s_intercept, zp_intercept = quantize_parameters(model)
    dequantize_and_infer(q_coef, q_intercept, s_coef, zp_coef, s_intercept, zp_intercept, scaler)

