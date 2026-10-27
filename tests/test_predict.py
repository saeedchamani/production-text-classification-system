from src.predict import predict_text


def test_predict_text():

    result = predict_text(
        "I forgot my password"
    )

    assert isinstance(result, dict)

    assert "intent" in result
    assert "category" in result
    assert "confidence" in result

    assert 0 <= result["confidence"] <= 1
    