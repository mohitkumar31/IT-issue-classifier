import gradio as gr
import requests
import chromadb

# Initialize ChromaDB client
client = chromadb.PersistentClient("data_embeddings")
collection = client.get_collection(name="common_problems_embeddings")

GOOGLE_API_KEY = "AIzaSyA6IPme4-PKOm1ggJkGQ1R16tKxeqk3rQQ"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GOOGLE_API_KEY}"

def chat_bot(message, history):
    # Query ChromaDB
    results = collection.query(
        query_texts=[message],
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )

    # Grab the metadata dict (assuming "Solution" is a key)
    metadata = results['metadatas'][0][0] if results['metadatas'] else {}

    # Prepare plain solution text if exists
    context = metadata.get("Solution", "")

    if not context:
        return "No relevant context found."

    # Gemini Prompt — only give it the question and clean context
    prompt = f"""
You are a helpful IT support assistant.

Here is the user's question:
{message}

And here is a possible solution from a knowledge base:
{context}

If the solution is relevant, return ONLY the plain answer without extra formatting or repetition.
If not relevant, just say "Sorry 😔 \n No relevant context found in our database".
"""

    headers = {"Content-Type": "application/json"}
    payload = {
        "generation_config": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 150,
            "response_mime_type": "text/plain"
        },
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return text.strip()
    else:
        return "Sorry, there was an error contacting the Gemini API."

# Chat interface
gr.ChatInterface(
    fn=chat_bot,
    title="IT Help Assistant 💬",
    description="Ask your IT questions and get helpful answers!",
    theme="soft"
).launch()