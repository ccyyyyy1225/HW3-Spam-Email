# predict.py
import joblib
import numpy as np

def explain_linear(text, vectorizer, model, top_k=10):
    X = vectorizer.transform([text])
    coef = model.coef_[0]
    feature_names = np.array(vectorizer.get_feature_names_out())
    contrib = X.toarray()[0] * coef
    idx = np.argsort(contrib)[::-1][:top_k]
    return list(zip(feature_names[idx], contrib[idx]))

def predict_email(text):
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    pred = model.predict(vectorizer.transform([text]))[0]
    explanation = explain_linear(text, vectorizer, model)
    return pred, explanation

if __name__ == "__main__":
    text = input("Enter an email: ")
    pred, explanation = predict_email(text)
    print("Prediction:", "Spam" if pred == 1 else "Ham")
    print("Top contributing tokens:")
    for token, weight in explanation:
        print(f"{token}: {weight:.4f}")
