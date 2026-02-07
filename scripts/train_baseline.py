import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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
    ngram_range=(1, 2),
    max_features=5000
)


X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42
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


def compute_risk_score(text, model, vectorizer):
    mots_urgents =["urgent", "verify", "now", "click"]
    reasons = []
    text = vectorizer.transform([text])
    score_rules, reasons = urgent_words_rules(text)
    probas = model.predict_proba(text)[0] 
    classes = list(model.classes_)
    if "phishing" in classes:
        idx = classes.index("phishing")
        score_ml = probas[idx] * 100
    else:
        score_ml = 0
    score_final = min(score_ml*0.7 + score_rules*0.3, 100)

    return score_final, reasons


def urgent_words_rules(text):
    mots_urgents = ["urgent", "verify", "now", "click", "confirm", "action required", "password", "billing"]
    score_rules = 0
    reasons = []
    text = text.lower()
    for x in mots_urgents:
        if x in text:
            score_rules+=20
            reasons.append(x)
    return score_rules, reasons
