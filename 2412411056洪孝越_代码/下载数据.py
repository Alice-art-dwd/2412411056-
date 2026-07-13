import pandas as pd

df = pd.read_parquet(r'C:\Users\HUAWEI\Desktop\test-00000-of-00001 (2).parquet')


df.to_csv(r'C:\Users\HUAWEI\Desktop\test1_50m.csv', index=False)