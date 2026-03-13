import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils import compute_risk_score

def load_datasets():
    # Charger la dataset
    all_emails = pd.read_csv("data/processed/all_emails.csv", sep=",", quotechar='"', engine="python")
    return all_emails

df = load_datasets()

df = pd.concat([df, df[df["label"] == "spam"]], ignore_index=True)

print(df["label"].value_counts())

# X = données textuelles (features)
X_text = df["text"]

# y = cible (labels)
y = df["label"]


vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000,
    stop_words="english",
    min_df=2
)


X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --- Fit TF-IDF sur le train uniquement ---
X_train_tfidf = vectorizer.fit_transform(X_train_text)

# --- Transformer le test ---
X_test_tfidf = vectorizer.transform(X_test_text)

# --- Vérifier que toutes les classes sont présentes ---
print("Classes dans le train :", set(y_train))
print("Classes dans le test  :", set(y_test))


print(X_train_tfidf.shape)
print(X_test_tfidf.shape)
print(vectorizer.get_feature_names_out())

model = LogisticRegression(max_iter=500,
                           solver='lbfgs')

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))

probas = model.predict_proba(X_test_tfidf)
print(probas)
print(model.classes_)





print(df["label"].value_counts())
print(df.isna().sum())


def extract_basic_features(text):

    text_lower = text.lower()

    features = {
        "length": len(text),
        "num_words": len(text.split()),
        "num_exclam": text.count("!"),
        "num_links": text_lower.count("http"),
        "num_digits": sum(c.isdigit() for c in text),
    }

    return features


# vectorizer = TfidfVectorizer(
#     ngram_range=(1,2),
#     max_features=5000,
#     stop_words="english",
#     min_df=2
# )




import joblib

joblib.dump(model, "models/phishing_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")


test_email = "Urgent: verify your account now by clicking this link"

score, reasons = compute_risk_score(test_email, model, vectorizer)

print("Test email :", test_email)
print("Risk score :", score)
print("Reasons :", reasons)


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))