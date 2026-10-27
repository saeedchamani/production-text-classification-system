from pathlib import Path

import joblib


MODEL_PATH = Path("models/text_classifier.joblib")


artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
intent_to_category = artifact["intent_to_category"]


def predict_text(text: str) -> dict:

    probabilities = pipeline.predict_proba([text])[0]

    classes = pipeline.classes_

    best_index = probabilities.argmax()

    intent = classes[best_index]

    confidence = float(probabilities[best_index])

    category = intent_to_category[intent]

    return {
        "intent": intent,
        "category": category,
        "confidence": round(confidence, 4),
    }


if __name__ == "__main__":

    text = "my payment is not working"

    result = predict_text(text)

    print(result)


    