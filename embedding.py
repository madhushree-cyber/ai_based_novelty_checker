import pandas as pd
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("AI_Project_Dataset_150.csv")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Combine useful text
df["combined"] = (
    df["Title"] + ". " +
    df["Description"] + ". " +
    df["Domain"] + ". " +
    df["Technologies"] + ". " +
    df["Features"] + ". " +
    df["Keywords"]
)

print("Generating embeddings...")

embeddings = model.encode(
    df["combined"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Done!")
print(embeddings.shape)