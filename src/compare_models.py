import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = "data/processed/cleaned_support_messages.csv"


df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["intent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
    ),

    "naive_bayes": MultinomialNB(),

    "linear_svm": LinearSVC(),
}


results = []


for model_name, model in models.items():

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
            model,
        ),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "f1_score": f1,
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False,
)

print("\nDataset size:")
print(len(df))

print("\nTrain size:")
print(len(X_train))

print("\nTest size:")
print(len(X_test))

print("\nModel comparison:")
print(results_df.to_string(index=False))