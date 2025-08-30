import pandas as pd

# Read the CSV file
df = pd.read_csv('vt_merged.csv')

print(df.columns)
print(df.head(20))

