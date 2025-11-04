# train.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from preprocessing import load_data, preprocess

df = load_data("dataset/sms_spam_no_header.csv")
X_train, X_test, y_train, y_test, vectorizer = preprocess(df)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
