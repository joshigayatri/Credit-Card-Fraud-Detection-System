import pandas as pd

df = pd.read_csv("data/creditcard.csv")

sample = df.head(100)

sample.to_csv(
    "sample_transactions.csv",
    index=False
)