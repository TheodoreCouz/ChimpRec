import pandas as pd

df = pd.read_csv("/home/theo/Documents/Unif/Master/ChimpRec/ChimpRec-Dataset/chimpbehave/labels.csv")

print(df.loc[df["class_id"] ==6])