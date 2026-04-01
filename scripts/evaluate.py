import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    cross_validate
)
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


def build_pipeline():
    """Crée le pipeline TF-IDF + Logistic Regression."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=500, solver='lbfgs'))
    ])


def evaluate_with_cross_validation(X, y, n_splits=5):
    """
    Évalue le modèle avec validation croisée stratifiée.
    Retourne le pipeline entraîné sur toutes les données.
    """
    pipeline = build_pipeline()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Scores F1 par fold
    scores_f1 = cross_val_score(
        pipeline, X, y,
        cv=skf,
        scoring='f1_weighted'
    )

    print("=== Validation croisée stratifiée ===")
    print(f"F1 par fold  : {scores_f1}")
    print(f"F1 moyen     : {scores_f1.mean():.3f}")
    print(f"Ecart-type   : {scores_f1.std():.3f}")

    # Détection overfitting
    results = cross_validate(
        pipeline, X, y,
        cv=skf,
        scoring=['f1_weighted', 'accuracy'],
        return_train_score=True
    )
    print(f"\nF1 train moyen : {results['train_f1_weighted'].mean():.3f}")
    print(f"F1 test moyen  : {results['test_f1_weighted'].mean():.3f}")
    if results['train_f1_weighted'].mean() - results['test_f1_weighted'].mean() > 0.1:
        print("⚠️  Attention : possible overfitting !")

    # Entraînement final sur toutes les données
    pipeline.fit(X, y)
    return pipeline


def plot_confusion_matrix(X, y, pipeline):
    """Affiche la matrice de confusion via validation croisée."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred_cv = cross_val_predict(pipeline, X, y, cv=skf)

    labels = ['legit', 'spam', 'phishing']
    cm = confusion_matrix(y, y_pred_cv, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title("Matrice de confusion — validation croisée stratifiée")
    plt.tight_layout()
    plt.show()

    print("\n=== Rapport de classification global ===")
    print(classification_report(y, y_pred_cv))
