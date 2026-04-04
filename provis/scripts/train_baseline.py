# scripts/train_baseline_final.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack
import joblib
from utils import compute_risk_score, urgent_words_rules

# ------------------------------
# 1. Charger les données
# ------------------------------
df = pd.read_csv("data/processed/all_emails.csv", sep=",", quotechar='"', engine="python")

# ------------------------------
# 2. Préparer le texte
# ------------------------------
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')
df['text'] = df['subject'] + ' ' + df['body']

# Nettoyage simple
df['text'] = df['text'].str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)
df = df[df['text'].str.strip() != '']  # supprimer textes vides

# ------------------------------
# 3. Features basiques
# ------------------------------
def extract_basic_features(text):
    text_lower = text.lower()
    return {
        "length": len(text),
        "num_words": len(text.split()),
        "num_exclam": text.count("!"),
        "num_links": text_lower.count("http"),
        "num_digits": sum(c.isdigit() for c in text),
    }

# ------------------------------
# 4. Séparer features et labels
# ------------------------------
X = df['text']
y = df['label']

# ------------------------------
# 5. Split train/test
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 6. TF-IDF vectorization
# ------------------------------
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ------------------------------
# 7. Extraire features basiques
# ------------------------------
basic_train = pd.DataFrame([extract_basic_features(t) for t in X_train])
basic_test = pd.DataFrame([extract_basic_features(t) for t in X_test])

# ------------------------------
# 8. Combiner TF-IDF + features basiques
# ------------------------------
X_train_combined = hstack([X_train_tfidf, basic_train.values])
X_test_combined = hstack([X_test_tfidf, basic_test.values])

# ------------------------------
# 9. Modèle Logistic Regression
# ------------------------------
clf = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',  # multiclass supporté
    random_state=42
)
clf.fit(X_train_combined, y_train)

# ------------------------------
# 10. Évaluation
# ------------------------------
y_pred = clf.predict(X_test_combined)

print("=== Classification Report ===\n")
print(classification_report(y_test, y_pred))

print("=== Matrice de Confusion ===\n")
print(confusion_matrix(y_test, y_pred))

# ------------------------------
# 11. Sauvegarde modèle et vectorizer
# ------------------------------
joblib.dump(clf, "models/phishing_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
print("\nModèle et vectorizer sauvegardés dans le dossier 'models/'")

# ------------------------------
# 12. Prédiction individuelle avec compute_risk_score
# ------------------------------
sample_email = "Urgent: verify your account now by clicking this link"

# Créer la combinaison TF-IDF + features basiques pour le sample
sample_vec = vectorizer.transform([sample_email.lower()])
sample_basic = pd.DataFrame([extract_basic_features(sample_email)])
sample_combined = hstack([sample_vec, sample_basic.values])

# Utiliser compute_risk_score pour obtenir le score et les raisons
score, reasons = compute_risk_score(sample_email, clf, vectorizer)

print("\n=== Test Email ===")
print(f"Email : {sample_email}")
print(f"Risk score : {score}")
print(f"Reasons : {reasons}")