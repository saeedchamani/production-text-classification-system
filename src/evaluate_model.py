import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = "data/processed/cleaned_support_messages.csv"


df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["intent"]


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
        LinearSVC(),
    ),
])


# -------------------------
# Cross Validation
# -------------------------

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1,
)

print("Cross Validation F1 scores:")
print(cv_scores)

print("\nMean CV F1:")
print(cv_scores.mean())

print("\nCV Standard Deviation:")
print(cv_scores.std())


# -------------------------
# Final train/test evaluation
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)


print("\nAccuracy:")
print(accuracy_score(y_test, predictions))


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
    )
)
