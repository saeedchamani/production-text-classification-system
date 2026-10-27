# Production Text Classification System

یک سامانهٔ طبقه‌بندی متن مبتنی بر Machine Learning برای تشخیص Intent پیام‌های پشتیبانی مشتریان.

این پروژه یک Pipeline کامل از دریافت Dataset تا ارائهٔ مدل از طریق REST API را پیاده‌سازی می‌کند.

## Features

- Multi-class text classification
- 27 customer-support intents
- TF-IDF text representation
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Model comparison
- Confidence score
- FastAPI REST API
- Input validation
- Automated testing with Pytest
- Prediction latency benchmark
- Docker support

## Dataset

این پروژه از دیتاست عمومی Bitext Customer Support استفاده می‌کند.

دیتاست خام شامل حدود:

```text
26,872 samples
27 intents
```

است.

پس از پاک‌سازی و حذف داده‌های تکراری و ناسازگار، حدود:

```text
24,014 samples
```

برای آموزش مدل باقی مانده است.

## Example Intents

```text
cancel_order
payment_issue
recover_password
track_order
get_refund
complaint
create_account
delete_account
delivery_period
contact_human_agent
```

## Machine Learning Pipeline

```text
Raw Dataset
    ↓
Data Cleaning
    ↓
Text Preprocessing
    ↓
TF-IDF
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Calibrated Linear SVM
    ↓
Saved Model
    ↓
FastAPI
    ↓
Prediction API
```

## Model Comparison

نتایج اولیه روی Test Set:

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Linear SVM | 0.9931 | 0.9931 |
| Logistic Regression | 0.9917 | 0.9917 |
| Naive Bayes | 0.9848 | 0.9848 |

مدل `LinearSVC` بهترین عملکرد اولیه را داشت.

برای تولید Confidence Score، مدل نهایی با `CalibratedClassifierCV` کالیبره شده است.

## Prediction Example

ورودی:

```text
I forgot my password
```

خروجی نمونه:

```json
{
  "intent": "recover_password",
  "category": "ACCOUNT",
  "confidence": 0.95
}
```

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
│   ├── api.py
│   ├── predict.py
│   ├── prepare_dataset.py
│   ├── compare_models.py
│   ├── evaluate_model.py
│   ├── train_final_model.py
│   └── benchmark_latency.py
│
├── tests/
│   ├── test_api.py
│   └── test_predict.py
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

## Installation

ابتدا Repository را دریافت کنید و وارد پوشهٔ پروژه شوید.

سپس یک محیط مجازی ایجاد کنید.

```bash
python -m venv .venv
source .venv/bin/activate
```

سپس وابستگی‌ها را نصب کنید.

```bash
python -m pip install -r requirements.txt
```

## Run API

برای اجرای FastAPI:

```bash
uvicorn src.api:app --reload
```

سپس Swagger UI در آدرس زیر قابل دسترسی است:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```text
GET /health
```

خروجی:

```json
{
  "status": "ok"
}
```

### Predict

```text
POST /predict
```

ورودی:

```json
{
  "text": "I forgot my password"
}
```

خروجی:

```json
{
  "intent": "recover_password",
  "category": "ACCOUNT",
  "confidence": 0.95
}
```

## Testing

برای اجرای تست‌ها:

```bash
python -m pytest -v
```

تست‌های فعلی شامل موارد زیر هستند:

```text
API health check
Prediction endpoint
Empty input validation
Missing input validation
Direct model prediction
```

## Performance

Benchmark محلی Prediction:

```text
Runs: 1000
Average latency: 3.200 ms
Min latency: 3.117 ms
Max latency: 3.745 ms
```

این اعداد مربوط به زمان اجرای Prediction مدل روی سیستم توسعه هستند و شامل Network Latency نمی‌شوند.

## Docker

برای ساخت Docker Image:

```bash
docker build -t text-classification-api .
```

برای اجرای Container:

```bash
docker run --rm \
  -p 8000:8000 \
  --name text-classifier \
  text-classification-api
```

سپس API از طریق آدرس زیر قابل دسترسی است:

```text
http://127.0.0.1:8000
```

و Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Tech Stack

```text
Python
Pandas
NumPy
Scikit-learn
TF-IDF
Linear SVM
FastAPI
Pydantic
Pytest
Docker
Git
```

## Notes

این پروژه با هدف تمرین چرخهٔ کامل توسعهٔ یک سیستم Machine Learning تولیدی ساخته شده است.

تمرکز پروژه فقط روی دقت مدل نیست، بلکه روی مواردی مانند ساختار پروژه، تست خودکار، API، Validation، Latency و Containerization نیز قرار دارد.