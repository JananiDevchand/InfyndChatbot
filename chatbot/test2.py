from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from collections import defaultdict
import os
import json

load_dotenv()
app = Flask(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "infyndcompanydata")
index = pc.Index(index_name)

# Embedding model & LLM
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = Ollama(model="phi3:mini")

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    query = request.form.get("msg", "").strip()
    if not query:
        return jsonify({"summary": "Please enter a query.", "structured_output": {}})

    print(f"🔍 Query: {query}")

    # Query Pinecone
    query_vector = embedder.encode(query).tolist()
    result = index.query(vector=query_vector, top_k=20, include_metadata=True)

    # Group and parse structured data
    grouped = defaultdict(list)
    for match in result.matches:
        meta = match.metadata or {}
        source = meta.get("source", "unknown_source")
        text = meta.get("text", "")
        if text:
            try:
                grouped[source].append(json.loads(text))
            except:
                grouped[source].append({"text": text})

    structured_output = dict(grouped)

    # Ask LLM for summary only
    prompt = f"""
You are a concise assistant. Summarize the following company information in 2-3 lines conversationally.

Query: "{query}"

Data:
{json.dumps(structured_output, indent=2)}

Do not include the JSON data or filters in your response.
    """

    try:
        summary = llm.invoke(prompt).strip()
    except Exception as e:
        summary = f"Error generating summary: {e}"

    # Return both structured data and LLM summary
    return jsonify({"summary": summary, "structured_output": structured_output})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
