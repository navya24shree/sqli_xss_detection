import joblib

model = joblib.load("attack_classifier.pkl")
tfidf = joblib.load("vectorizer.pkl")

def predict_payload(text):
    vector = tfidf.transform([text])
    result = model.predict(vector)[0]
    return result

# Test
print(predict_payload("<script>alert(1)</script>"))
