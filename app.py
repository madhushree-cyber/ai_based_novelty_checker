import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from groq_analysis import analyze_project

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Project Novelty Checker",
    page_icon="🧠",
    layout="wide"
)

DATA_PATH = "AI_Project_Dataset_150.csv"


# ----------------------------------------------------------------------
# Cached resources — loaded once per session, shared across reruns
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["combined"] = (
        df["Title"] + ". " +
        df["Description"] + ". " +
        df["Domain"] + ". " +
        df["Technologies"] + ". " +
        df["Features"] + ". " +
        df["Keywords"]
    )
    return df


@st.cache_resource(show_spinner=False)
def build_index(_model, combined_texts):
    embeddings = _model.encode(combined_texts, show_progress_bar=False)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(embeddings)
    return nn, embeddings


def get_similar_projects(query, df, model, nn, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = nn.kneighbors(query_embedding, n_neighbors=top_k)

    results = []
    similar_projects_text = ""

    for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
        similarity = (1 - distance) * 100
        row = df.iloc[index]
        results.append({
            "rank": i + 1,
            "title": row["Title"],
            "domain": row["Domain"],
            "description": row["Description"],
            "technologies": row["Technologies"],
            "similarity": similarity
        })
        similar_projects_text += f"""
Project {i+1}

Title: {row['Title']}
Description: {row['Description']}
Domain: {row['Domain']}
Technologies: {row['Technologies']}
Features: {row['Features']}
Similarity: {similarity:.2f}%

"""
    return results, similar_projects_text


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Novelty Checker")
    st.markdown(
        "Describe your AI project idea and get:\n"
        "- The most similar existing projects\n"
        "- A novelty score\n"
        "- Suggestions to make it more original"
    )
    st.divider()

    if not os.getenv("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY not found. Set it in a `.env` file before running.")

    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset `{DATA_PATH}` not found in the app folder.")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------------------------------------
# Load data + model + index (cached, so this is fast after first run)
# ----------------------------------------------------------------------
if os.path.exists(DATA_PATH):
    with st.spinner("Loading project database and embedding model..."):
        df = load_data()
        model = load_model()
        nn, embeddings = build_index(model, df["combined"].tolist())
else:
    st.stop()


# ----------------------------------------------------------------------
# Chat state
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("AI Project Novelty Checker")
st.caption("Chat-style interface for evaluating how novel your AI project idea is.")

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "similar" in msg:
            with st.expander("🔍 Top similar projects", expanded=False):
                for proj in msg["similar"]:
                    st.markdown(
                        f"**{proj['rank']}. {proj['title']}**  "
                        f"({proj['similarity']:.2f}% similar)\n\n"
                        f"*Domain:* {proj['domain']}  \n"
                        f"*Technologies:* {proj['technologies']}  \n"
                        f"{proj['description']}"
                    )
                    st.markdown("---")
        st.markdown(msg["content"])


# ----------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------
query = st.chat_input("Describe your project idea...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching similar projects..."):
            results, similar_projects_text = get_similar_projects(query, df, model, nn)

        with st.expander("🔍 Top similar projects", expanded=False):
            for proj in results:
                st.markdown(
                    f"**{proj['rank']}. {proj['title']}**  "
                    f"({proj['similarity']:.2f}% similar)\n\n"
                    f"*Domain:* {proj['domain']}  \n"
                    f"*Technologies:* {proj['technologies']}  \n"
                    f"{proj['description']}"
                )
                st.markdown("---")

        with st.spinner("Analyzing novelty with Groq..."):
            try:
                analysis = analyze_project(query, similar_projects_text)
            except Exception as e:
                analysis = f"⚠️ Error while calling Groq API: {e}"

        st.markdown(analysis)

    st.session_state.messages.append({
        "role": "assistant",
        "content": analysis,
        "similar": results
    })