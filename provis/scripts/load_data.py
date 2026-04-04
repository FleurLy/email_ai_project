import pandas as pd
# import matplotlib.pyplot as plt

def load_raw_datasets():
    # Charger chaque CSV
    # phishing = pd.read_csv("data/raw/phishing.csv", sep=";", quotechar='"', engine="python")
    # spam = pd.read_csv("data/raw/spam.csv", sep=";", quotechar='"', engine="python")
    # legit = pd.read_csv("data/raw/legit.csv", sep=";", quotechar='"', engine="python")
    # legit_urgent = pd.read_csv("data/raw/legit_urgent.csv", sep=";", quotechar='"', engine="python")
    # spam_long = pd.read_csv("data/raw/spam_long.csv", sep=";", quotechar='"', engine="python")
    # phishing_short = pd.read_csv("data/raw/phishing_short.csv", sep=";", quotechar='"', engine="python")
    # legit_full = pd.read_csv("data/raw/legit_full.csv", sep=";", quotechar='"', engine="python")
    # spam_full = pd.read_csv("data/raw/spam_full.csv", sep=";", quotechar='"', engine="python")
    # phishing_full = pd.read_csv("data/raw/phishing_full.csv", sep=";", quotechar='"', engine="python")
    emails_1 = pd.read_csv("data/raw/emails_1.csv", sep=",", quotechar='"', engine="python")
    emails_2 = pd.read_csv("data/raw/emails_2.csv", sep=",", quotechar='"', engine="python")
    emails_3 = pd.read_csv("data/raw/emails_3.csv", sep=",", quotechar='"', engine="python")
    emails_4 = pd.read_csv("data/raw/emails_4.csv", sep=",", quotechar='"', engine="python")
    emails_5 = pd.read_csv("data/raw/emails_5.csv", sep=",", quotechar='"', engine="python")

    # return phishing, spam, legit, legit_urgent, spam_long, phishing_short, legit_full, spam_full, phishing_full, 
    return emails_1, emails_2, emails_3, emails_4, emails_5


if __name__ == "__main__":
    # phishing, spam, legit, legit_urgent, spam_long, phishing_short, legit_full, spam_full, phishing_full, 
    emails_1, emails_2, emails_3, emails_4, emails_5 = load_raw_datasets()
    # print("Phishing:", len(phishing))
    # print("Spam:", len(spam))
    # print("Legit:", len(legit))
    # print("Legit_urgent:", len(legit_urgent))
    # print("Spam_long:", len(spam_long))
    # print("Phishing_short:", len(phishing_short))
    # print("Legit_full:", len(legit_full))
    # print("Spam_full:", len(spam_full))
    # print("Phishing_full:", len(phishing_full))
    print("Emails_1:", len(emails_1))
    print("Emails_2:", len(emails_2))
    print("Emails_3:", len(emails_3))
    print("Emails_4:", len(emails_4))
    print("Emails_5:", len(emails_5))
