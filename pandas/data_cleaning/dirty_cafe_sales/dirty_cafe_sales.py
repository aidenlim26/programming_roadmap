import pandas as pd
import numpy as np

df = pd.read_csv("dirty_cafe_sales.csv")

#CLEAN HEADERS
#print(df.columns.to_list())
df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")
#print("FIX APPLIED")
#print(df.columns.to_list())

#REMOVING DIRTY ROWS
df = df.replace(["ERROR","UNKNOWN"],np.nan,regex=True)
#print(df.isna().sum())
df = df.dropna()
print(df.to_string())