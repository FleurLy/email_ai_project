from prepare_data import prepare_dataset
from evaluate import evaluate_with_cross_validation, plot_confusion_matrix
from scoring import compute_risk_score

# 1. Charger et préparer les données
df = prepare_dataset()
X_text = df["text"]
y = df["label"]

# 2. Évaluer avec validation croisée stratifiée
pipeline = evaluate_with_cross_validation(X_text, y, n_splits=5)

# 3. Matrice de confusion
plot_confusion_matrix(X_text, y, pipeline)

# 4. Tester le scoring hybride
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
    score, reasons = compute_risk_score(
        text,
        pipeline.named_steps['clf'],
        pipeline.named_steps['tfidf']
    )
    print(f"Email  : {text}")
    print(f"Score  : {score:.1f}/100")
    print(f"Raisons: {reasons}\n")
