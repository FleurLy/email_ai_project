import pandas as pd
from load_data import load_raw_datasets

def prepare_dataset():
    phishing, spam, legit, legit_urgent, spam_long, phishing_short, legit_full, spam_full, phishing_full  = load_raw_datasets()

    phishing["label"] = "phishing"
    spam["label"] = "spam"
    legit["label"] = "legit"

    legit_urgent["label"] = "legit_urgent"
    spam_long["label"] = "spam_long"
    phishing_short["label"] = "phishing_short"

    legit_full["label"] = "legit_full"
    spam_full["label"] = "spam_full"
    phishing_full["label"] = "phishing_full"

    df = pd.concat([phishing, spam, legit, legit_urgent, spam_long, phishing_short, legit_full, spam_full, phishing_full], ignore_index=True)

    df.drop_duplicates(inplace=True)
    df.dropna(subset=["text"], inplace=True)

    return df

if __name__ == "__main__":
    df = prepare_dataset()
    print("Emails par classe:")
    print(df["label"].value_counts())

    df.to_csv("data/processed/all_emails.csv", index=False)
    print("Dataset sauvegardé dans data/processed/all_emails.csv")
