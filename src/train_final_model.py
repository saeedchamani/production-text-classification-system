from pathlib import Path

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = Path("data/processed/cleaned_support_messages.csv")
MODEL_PATH = Path("models/text_classifier.joblib")


df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["intent"]


intent_to_category = (
    df[["intent", "category"]]
    .drop_duplicates()
    .set_index("intent")["category"]
    .to_dict()
)


classifier = CalibratedClassifierCV(
    estimator=LinearSVC(),
    cv=5,
)


pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        classifier,
    ),
])


print("Training final model...")

pipeline.fit(X, y)


artifact = {
    "pipeline": pipeline,
    "intent_to_category": intent_to_category,
    "model_name": "LinearSVC + CalibratedClassifierCV",
    "dataset_size": len(df),
}


MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


joblib.dump(
    artifact,
    MODEL_PATH,
)


print(f"Model saved to: {MODEL_PATH}")
print(f"Dataset size: {len(df)}")
print(f"Number of intents: {y.nunique()}")