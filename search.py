import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from groq_analysis import analyze_project

# Load dataset
df = pd.read_csv("AI_Project_Dataset_150.csv")

# Combine useful information
df["combined"] = (
    df["Title"] + ". " +
    df["Description"] + ". " +
    df["Domain"] + ". " +
    df["Technologies"] + ". " +
    df["Features"] + ". " +
    df["Keywords"]
)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(df["combined"].tolist())

print("Training search model...")
nn = NearestNeighbors(
    n_neighbors=5,
    metric="cosine"
)

nn.fit(embeddings)

print("Ready!")

query = input("\nEnter your project idea: ")

query_embedding = model.encode([query])

distances, indices = nn.kneighbors(query_embedding)

print("\nTop 5 Similar Projects:\n")

for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
    similarity = (1 - distance) * 100

    print(f"{i+1}. {df.iloc[index]['Title']}")
    print(f"Similarity: {similarity:.2f}%")
    print(f"Domain: {df.iloc[index]['Domain']}")
    print("-" * 40)
    
similar_projects = ""

for i, (distance, index) in enumerate(zip(distances[0], indices[0])):

    similarity = (1-distance)*100

    similar_projects += f"""
Project {i+1}

Title: {df.iloc[index]['Title']}
Description: {df.iloc[index]['Description']}
Domain: {df.iloc[index]['Domain']}
Technologies: {df.iloc[index]['Technologies']}
Features: {df.iloc[index]['Features']}
Similarity: {similarity:.2f}%

"""

analysis = analyze_project(query, similar_projects)

print("\n")
print("="*80)
print(analysis)