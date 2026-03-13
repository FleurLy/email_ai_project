def compute_risk_score(text, model, vectorizer):

    #--- règles sur texte brut ---
    score_rules, reasons = urgent_words_rules(text)

    # --- score ML ---
    text_vec = vectorizer.transform([text])
    probas = model.predict_proba(text_vec)[0]

    classes = list(model.classes_)

    if "phishing" in classes:
        idx = classes.index("phishing")
        score_ml = probas[idx] * 100
    else:
        score_ml = 0

    score_final = min(score_ml * 0.7 + score_rules * 0.3, 100)

    return score_final, reasons


def urgent_words_rules(text):

    mots_urgents = [
        "urgent", "verify", "now", "click",
        "confirm", "action required",
        "password", "billing", "login",
        "security alert"
    ]

    score_rules = 0
    reasons = []

    text = text.lower()

    for word in mots_urgents:
        if word in text:
            score_rules += 20
            reasons.append(word)

    return score_rules, reasons