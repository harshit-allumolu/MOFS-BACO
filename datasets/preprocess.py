import pandas as pd

df = pd.read_csv("vehicle.csv",header=None)

for x in df.columns:
    if x!=0:
        df[x] = (df[x]-min(df[x]))/(max(df[x]-min(df[x])))

# print(df)
df.to_csv("vehicle.csv",index=None,header=None)