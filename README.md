# MLOps Linear Regression Pipeline

This repository contains an end-to-end MLOps pipeline for a Linear Regression model, focusing on training, testing, quantization, Dockerization, and CI/CD integration. The project utilizes the California Housing dataset and scikit-learn's LinearRegression model.




## Project Introduction

This project implements a complete MLOps pipeline for a Linear Regression model. The pipeline covers data loading, model training, model quantization, and deployment readiness through Dockerization. It also includes a basic CI/CD workflow setup to ensure continuous integration and delivery practices.

**Key Features:**
*   **Linear Regression Model:** Utilizes scikit-learn's LinearRegression for predictive modeling.
*   **California Housing Dataset:** Employs a standard dataset for regression tasks.
*   **Model Quantization:** Demonstrates a manual quantization process to reduce model size, crucial for efficient deployment in resource-constrained environments.
*   **Dockerization:** Provides a `Dockerfile` for containerizing the application, ensuring consistent environments across development, testing, and production.
*   **CI/CD Integration:** Includes configurations for a CI/CD workflow to automate testing and deployment processes.

---

## Directory Structure

```txt
mlops-linear-regression-pipeline/
├── Dockerfile
├── README.md
├── requirements.txt
├── src/
│   ├── train.py
│   ├── quantize.py
│   ├── predict.py
│   └── utils.py
├── test/
│   └── test_train.py
├── regression_model.joblib
├── unquant_params.joblib
├── quant_params.joblib

```

*   `Dockerfile`: Defines the Docker image for the application.
*   `README.md`: This file, providing an overview and instructions.
*   `quant_params.joblib`: Quantized model parameters.
*   `regression_model.joblib`: The trained Linear Regression model.
*   `unquant_params.joblib`: Unquantized model parameters.
*   `requirements.txt`: Lists all Python dependencies.
*   `src/`: Contains the source code for the pipeline components.
    *   `predict.py`: Script for making predictions.
    *   `quantize.py`: Script for quantizing the trained model.
    *   `train.py`: Script for training the Linear Regression model.
    *   `utils.py`: Utility functions.
*   `test/`: Contains unit tests for the pipeline.
    *   `test_train.py`: Tests for the training script.


## Project Setup

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shushant15/mlops-linear-regression-pipeline.git
    cd mlops-linear-regression-pipeline
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

After setting up the project, you can run the following scripts:

1.  **Train the Linear Regression Model:**
    This script trains the model and saves the `regression_model.joblib` and `unquant_params.joblib` files.
    ```bash
    python3 src/train.py
    ```
    Expected output will include the R² Score and Mean Squared Error.

2.  **Quantize the Trained Model:**
    This script quantizes the `unquant_params.joblib` and saves the `quant_params.joblib` file.
    ```bash
    python3 src/quantize.py
    ```
    Expected output will include sample predictions and the R² Score for the de-quantized model.

3.  **Run Tests:**
    To ensure everything is working correctly, you can run the provided tests using `pytest`.
    ```bash
    pytest test/test_train.py
    ```


## Model Comparison

The table below shows how the original scikit-learn Linear Regression model compares with the quantized version, using the actual file sizes from our project:

| Metric           | Original Sklearn Model<br>`unquant_params.joblib` | Quantized Model<br>`quant_params.joblib` |
|------------------|:-------------------------------------------------:|:----------------------------------------:|
| **R² Score**     | 0.5758                                            | 0.5754                                   |
| **Model Size**   | 0.404 KB                                          | 0.356 KB                                 |

**Analysis:**

- **R² Score:**  
  Both models perform almost identically, with only a tiny difference in the R² score (0.0004). This means the quantization process doesn’t noticeably affect the accuracy of our predictions, which is what we want.

- **Model Size:**  
  The quantized model (`quant_params.joblib`, 0.356 KB) is slightly smaller than the unquantized model (`unquant_params.joblib`, 0.404 KB). This is because we’ve stored the model weights using a lower-precision data type (`np.uint8`).  
  However, the size difference is quite small. This is mostly because the model itself is very small, and Python’s way of saving files (serialization) adds a bit of overhead. In much bigger models (like deep learning networks), quantization would have a much bigger impact.

- **Conclusion:**  
  In this assignment, we showed that quantization can reduce model size without losing accuracy. For small models like this, the effect is limited, but the technique is valuable for deploying large models to environments where storage and memory matter.

## Author

Name: Shushant Kumar Tiwari  
Roll No: G24AI1116  
Repo: https://github.com/shushant15/mlops-linear-regression-pipeline#