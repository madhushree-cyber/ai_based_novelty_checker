import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def analyze_project(query, similar_projects):

    prompt = f"""
You are an expert AI project evaluator.

A student has proposed the following project:

{query}

The following projects were retrieved from the project database because they are the most semantically similar.

{similar_projects}

Based on these projects, provide:

1. Novelty Score (0-100)
2. Similarities
3. Unique Aspects
4. Five Suggestions to Improve Novelty
5. Recommended Technologies
6. Future Scope
7. Final Verdict

Return the answer in markdown.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4
    )

    return response.choices[0].message.content