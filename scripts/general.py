import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Charger tous les fichiers CSV
raw_folder = "data/raw/"
csv_files = [
    "legit.csv",
    "phishing.csv",
    "spam.csv",
    "legit_urgent.csv",
    "phishing_short.csv",
    "spam_long.csv",
    "legit_full.csv",
    "phishing_full.csv",
    "spam_full.csv"
]

dfs = []
for file in csv_files:
    path = os.path.join(raw_folder, file)
    if os.path.exists(path):
        df = pd.read_csv(path, sep=";", quotechar='"', engine="python")
        #Définir la classe en fonction du nom de fichier
        if "legit" in file:
            df["label"] = "legit"
        elif "spam" in file:
            df["label"] = "spam"
        elif "phishing" in file:
            df["label"] = "phishing"
        dfs.append(df)
    else:
        print(f"Fichier manquant: {file}")

#Créer dataset combiné
all_emails = pd.concat(dfs, ignore_index=True)
all_emails.drop_duplicates(inplace=True)
all_emails.dropna(subset=["text"], inplace=True)

#Sauvegarder dataset final
os.makedirs("data/processed", exist_ok=True)
all_emails.to_csv("data/processed/all_emails.csv", index=False)
print("Dataset combiné sauvegardé dans data/processed/all_emails.csv")
print(all_emails["label"].value_counts())

#Préparer TF-IDF et modèle
X_text = all_emails["text"]
y = all_emails["label"]

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("\n=== Classification report sur le test set ===")
print(classification_report(y_test, y_pred))

#compute_risk_score & urgent_words_rules
def urgent_words_rules(text):
    mots_urgents = ["urgent", "verify", "now", "click", "confirm", "action required", "password", "billing", "login", "security alert"]
    score_rules = 0
    reasons = []
    text = text.lower()
    for x in mots_urgents:
        if x in text:
            score_rules += 20
            reasons.append(x)
    return score_rules, reasons

def compute_risk_score(text, model, vectorizer):
    X_email = vectorizer.transform([text])
    probas = model.predict_proba(X_email)[0]
    classes = list(model.classes_)
    if "phishing" in classes:
        idx = classes.index("phishing")
        score_ml = probas[idx] * 100
    else:
        score_ml = 0
    score_rules, reasons = urgent_words_rules(text)
    score_final = min(score_ml*0.7 + score_rules*0.3, 100)
    return score_final, reasons

#Tester quelques emails
examples = [
    "Please find attached the project report",
    "Win a free iPhone now!!! Limited offer",
    "Urgent: verify your account immediately by clicking this link",
    "Action required: reset your password immediately",
    "Lunch meeting tomorrow at 12pm",
    "Confirm your billing information to avoid account suspension"
]

print("\n=== Test compute_risk_score ===")
for text in examples:
    score, reasons = compute_risk_score(text, model, vectorizer)
    print(f"Email: {text}")
    print(f"Score: {score:.1f}/100, Reasons: {reasons}\n")
