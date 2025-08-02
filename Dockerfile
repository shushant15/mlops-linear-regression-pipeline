FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY regression_model.joblib .
COPY unquant_params.joblib .
COPY quant_params.joblib .

CMD ["python", "src/predict.py"]
