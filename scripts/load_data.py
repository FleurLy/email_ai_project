import pandas as pd
import matplotlib.pyplot as plt

def load_raw_datasets():
    # Charger chaque CSV
    phishing = pd.read_csv("data/raw/phishing.csv", sep=";", quotechar='"', engine="python")
    spam = pd.read_csv("data/raw/spam.csv", sep=";", quotechar='"', engine="python")
    legit = pd.read_csv("data/raw/legit.csv", sep=";", quotechar='"', engine="python")
    legit_urgent = pd.read_csv("data/raw/legit_urgent.csv", sep=";", quotechar='"', engine="python")
    spam_long = pd.read_csv("data/raw/spam_long.csv", sep=";", quotechar='"', engine="python")
    phishing_short = pd.read_csv("data/raw/phishing_short.csv", sep=";", quotechar='"', engine="python")

    return phishing, spam, legit, legit_urgent, spam_long, phishing_short


if __name__ == "__main__":
    phishing, spam, legit, legit_urgent, spam_long, phishing_short = load_raw_datasets()
    print("Phishing:", len(phishing))
    print("Spam:", len(spam))
    print("Legit:", len(legit))
    print("Legit_urgent:", len(legit_urgent))
    print("Spam_long:", len(spam_long))
    print("Phishing_short:", len(phishing_short))
