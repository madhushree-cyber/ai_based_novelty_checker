import pandas as pd 

df = pd.read_csv("AI_Project_Dataset_150.csv")
print("Dataset loaded successfully!")

print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print(f"\nTotal Projects: {len(df)}")