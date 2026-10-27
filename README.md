# Production Text Classification System

A production-oriented machine learning project for multi-class customer support intent classification.

The project covers the complete workflow from dataset preparation and model evaluation to REST API serving, automated testing, latency benchmarking, and Docker containerization.

## Overview

The system receives a customer support message and predicts:

- the most likely intent
- the corresponding high-level category
- a calibrated confidence score

Example input:

```json
{
  "text": "I forgot my password"
}
```

Example response:

```json
{
  "intent": "recover_password",
  "category": "ACCOUNT",
  "confidence": 0.95
}
```

## Features

- 27-class customer support intent classification
- Text preprocessing and dataset cleaning
- TF-IDF feature extraction
- Logistic Regression baseline
- Multinomial Naive Bayes baseline
- Linear SVM classification
- Model comparison using Accuracy and Weighted F1
- Probability calibration for confidence scores
- FastAPI REST API
- Pydantic request validation
- Automated API and prediction tests with Pytest
- Local inference latency benchmarking
- Dockerized API deployment
- Reproducible dataset download workflow

## Dataset

The project uses the public **Bitext Customer Support LLM Chatbot Training Dataset**.

The original dataset contains approximately:

```text
26,872 samples
27 intents
```

The classification target is the `intent` column, while the `category` column is used as a higher-level grouping.

After cleaning, duplicate removal, and conflict filtering, the processed dataset contains:

```text
24,014 samples
```

The raw dataset is not stored in this repository.

It can be downloaded using:

```bash
python -m src.download_dataset
```

The download script also verifies the SHA-256 checksum of the dataset.

### Dataset Note

The Bitext dataset contains synthetic and structured customer-support examples.

Because of its relatively consistent language patterns, evaluation scores on this dataset can be higher than what should be expected on noisy real-world customer conversations.

## Example Intents

The dataset includes intents such as:

```text
recover_password
payment_issue
cancel_order
track_order
track_refund
get_refund
complaint
create_account
delete_account
delivery_period
contact_human_agent
change_shipping_address
```

## Machine Learning Pipeline

```text
Raw Dataset
    ↓
Data Cleaning
    ↓
Text Normalization
    ↓
Train / Test Split
    ↓
TF-IDF Vectorization
    ↓
Model Training
    ↓
Model Comparison
    ↓
Linear SVM
    ↓
Probability Calibration
    ↓
Serialized Model
    ↓
FastAPI
    ↓
REST Prediction API
```

## Model Comparison

Three classical machine learning models were evaluated using the same train/test split.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Linear SVM | 0.9931 | 0.9931 |
| Logistic Regression | 0.9917 | 0.9917 |
| Multinomial Naive Bayes | 0.9848 | 0.9848 |

Linear SVM achieved the strongest result and was selected as the final classifier.

Since `LinearSVC` does not directly produce class probabilities, the final model uses `CalibratedClassifierCV` to provide confidence scores.

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── text_classifier.joblib
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── benchmark_latency.py
│   ├── compare_models.py
│   ├── download_dataset.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── prepare_dataset.py
│   └── train_final_model.py
│
├── tests/
│   ├── test_api.py
│   └── test_predict.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/saeedchamani/production-text-classification-system.git
cd production-text-classification-system
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install runtime and development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## Reproducing the Model

Download the dataset:

```bash
python -m src.download_dataset
```

Prepare the cleaned dataset:

```bash
python -m src.prepare_dataset
```

Compare candidate models:

```bash
python -m src.compare_models
```

Train and save the final model:

```bash
python -m src.train_final_model
```

The trained model is saved as:

```text
models/text_classifier.joblib
```

Model artifacts are excluded from Git because they are generated files.

## Running the API

Start the FastAPI application:

```bash
uvicorn src.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "text": "I forgot my password"
}
```

Example response:

```json
{
  "intent": "recover_password",
  "category": "ACCOUNT",
  "confidence": 0.95
}
```

Empty or missing input is rejected using FastAPI/Pydantic validation.

## Testing

Run the automated test suite:

```bash
python -m pytest -v
```

The test suite currently covers:

- API health endpoint
- prediction endpoint
- empty input validation
- missing input validation
- direct model prediction

Current result:

```text
5 passed
```

## Latency Benchmark

Local model inference was benchmarked over 1,000 prediction calls.

```text
Runs:             1000
Average latency:  3.200 ms
Minimum latency:  3.117 ms
Maximum latency:  3.745 ms
```

These measurements represent local model inference latency and do not include HTTP or network overhead.

The benchmark can be reproduced with:

```bash
python -m src.benchmark_latency
```

## Docker

Build the Docker image:

```bash
docker build -t text-classification-api .
```

Run the container:

```bash
docker run --rm \
  -p 8000:8000 \
  --name text-classifier \
  text-classification-api
```

The containerized API will be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Technology Stack

- Python 3.12
- NumPy
- Pandas
- Scikit-learn
- TF-IDF
- Linear SVM
- FastAPI
- Pydantic
- Uvicorn
- Pytest
- Docker
- Git
- GitHub

## Engineering Decisions

### TF-IDF and Linear SVM

TF-IDF combined with a linear classifier provides a strong and computationally efficient baseline for structured text classification problems.

### Confidence Calibration

`LinearSVC` produces decision scores rather than probabilities.

`CalibratedClassifierCV` is therefore used in the final model to produce probability-like confidence scores suitable for the API response.

### Model Serialization

The trained pipeline is stored with `joblib`, allowing the API to load the complete preprocessing and classification pipeline without retraining at startup.

### Docker

Docker packages the runtime environment, Python dependencies, application code, and trained model into a reproducible deployment environment.

## Limitations

The current model is trained primarily on structured English customer-support messages.

Performance on real-world conversations may decrease when messages contain:

- spelling mistakes
- slang
- highly ambiguous requests
- multiple intents in the same message
- languages not represented in the training dataset
- domain-specific terminology outside the training distribution

Future work can include evaluation on real-world data and transformer-based multilingual models.

