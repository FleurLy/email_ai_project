import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/all_emails.csv")

df["label"].value_counts().plot(
    kind="bar",
    title="Emails par classe"
)

plt.tight_layout()
plt.show()
