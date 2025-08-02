import pytest
import os
import sys
import joblib
import warnings
from sklearn.linear_model import LinearRegression
import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_data
from src.train import train_model

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestTraining:
    
    def test_dataset_loading(self):
        """Test that the dataset loads correctly."""
        X_train, X_test, y_train, y_test = load_data()
        
        # Check that data is loaded
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        
        # Check data shapes
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        
        # Check that features match
        assert X_train.shape[1] == X_test.shape[1]
        assert X_train.shape[1] == 8  # California housing has 8 features
    
    def test_model_creation(self):
        """Test that the model is created as a LinearRegression instance."""
        model, r2, mse = train_model()
        
        # Check that model is LinearRegression instance
        assert isinstance(model, LinearRegression)
    
    def test_model_trained(self):
        """Test that the model was trained (coefficients exist)."""
        model, r2, mse = train_model()
        
        # Check that coefficients exist (model was fitted)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None
        
        # Check coefficient shape matches number of features
        assert len(model.coef_) == 8
    
    def test_r2_threshold(self):
        """Test that R2 score exceeds minimum threshold."""
        model, r2, mse = train_model()
        
        # R2 score should be reasonable for this dataset (at least 0.5)
        assert r2 > 0.5
        
        # MSE should be positive
        assert mse > 0
    
    def test_model_saved(self):
        """Test that the model is saved correctly."""
        train_model()
        
        # Check that model file exists
        assert os.path.exists('regression_model.joblib')
        
        # Check that saved model can be loaded
        loaded_artifacts = joblib.load("regression_model.joblib")
        loaded_model = loaded_artifacts["model"]
        assert isinstance(loaded_model, LinearRegression)
        
        # Clean up
        if os.path.exists('regression_model.joblib'):
            os.remove('regression_model.joblib')
