import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/creditcard.csv")

df['Class'].value_counts().plot(kind='bar')

plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()