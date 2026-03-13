import joblib
from utils import compute_risk_score

model = joblib.load("models/phishing_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")



email = "Urgent verify your account now"


vec = vectorizer.transform([email])

print(model.predict(vec))
print(model.predict_proba(vec))

emails = [
    "Urgent verify your account now",
    "Security alert: confirm your password immediately",
    "You won a prize! click here to claim",
    "Hello, the meeting will take place tomorrow at 10am.",
    "Please find attached the report for your review.",
    "Your bank account has been suspended, login to resolve.",
]

for email in emails:
    print("\nEmail:", email)
    score, reasons = compute_risk_score(email, model, vectorizer)
    print("Risk score:", score)
    print("Reasons:", reasons)


print("Model classes:", model.classes_)
print("Vectorizer loaded correctly.")


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))