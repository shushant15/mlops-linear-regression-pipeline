import pytest
import os
import sys
import joblib
import warnings
from sklearn.linear_model import LinearRegression
import pytest
warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_data
from src.train import train_model

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestTraining:
    
    def test_dataset_loading(self):
        X_train, X_test, y_train, y_test = load_data()
        
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        
        assert X_train.shape[1] == X_test.shape[1]
        assert X_train.shape[1] == 8  
    
    def test_model_creation(self):
        model, r2, mse = train_model()
        
        assert isinstance(model, LinearRegression)
    
    def test_model_trained(self):
        model, r2, mse = train_model()
        
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None
        
        assert len(model.coef_) == 8
    
    def test_r2_threshold(self):
        model, r2, mse = train_model()
        
        assert r2 > 0.5
        
        assert mse > 0
    
    def test_model_saved(self):
        train_model()
        
        assert os.path.exists('regression_model.joblib')
        
        loaded_artifacts = joblib.load("regression_model.joblib")
        loaded_model = loaded_artifacts["model"]
        assert isinstance(loaded_model, LinearRegression)
        
        if os.path.exists('regression_model.joblib'):
            os.remove('regression_model.joblib')
