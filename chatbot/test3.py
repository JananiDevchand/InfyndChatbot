from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from collections import defaultdict
from json_repair import repair_json
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

# ---------------------------
# Utility: JSON repair + validation
# ---------------------------
def validate_and_repair_json(llm_output: str):
    """Attempts to parse or repair malformed JSON returned by the LLM."""
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        try:
            repaired = repair_json(llm_output)
            return json.loads(repaired)
        except Exception as e:
            raise ValueError(f"Failed to repair/parse JSON: {e}")

@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    query = request.form.get("msg", "").strip()
    if not query:
        return jsonify({
            "summary": "Please enter a query.",
            "validated_output": {},
            "elasticsearch_query": {}
        })

    print(f"\n🔍 Query: {query}")

    # ---- Step 1: Query Pinecone ----
    try:
        query_vector = embedder.encode(query).tolist()
        result = index.query(vector=query_vector, top_k=20, include_metadata=True)
    except Exception as e:
        print(f"❌ Pinecone error: {e}")
        return jsonify({
            "summary": f"Error connecting to database: {e}",
            "validated_output": {},
            "elasticsearch_query": {}
        })

    # ---- Step 2: Group results by source ----
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
    print(f"🔹 Structured output before validation: {structured_output}")

    # ---- Step 3: Build Prompt for LLM ----
    prompt = f"""
You are a precise JSON validator, summarizer, and Elasticsearch query generator.

Given the user's query and structured data grouped by source files:
1. Validate the JSON data — keep only relevant items for the user's query.
2. Generate a short conversational summary (2–3 lines).
3. Construct a valid Elasticsearch query based on the validated data, 
   similar to how 'company_curl1.txt' or 'people_curl1.txt' use filters.

The output must be a valid JSON (no extra text or code fences) in this format:
{{
  "validated_output": {{
      "Subindustry.json": [{{...}}],
      "company_type.json": [{{...}}],
      "job_description.json": [{{...}}],
      "cd_sicCode.json": [{{...}}]
  }},
  "summary": "A short conversational summary (2–3 lines).",
  "elasticsearch_query": {{
      "query": {{
          "bool": {{
              "must": [ ... ]
          }}
      }},
      "_source": [ ... ],
      "size": 10,
      "sort": [{{"company_score": {{"order": "desc"}}}}]
  }}
}}

User Query: "{query}"

Structured JSON:
{json.dumps(structured_output, indent=2)}
"""

    # ---- Step 4: Get & Parse LLM Output ----
    try:
        llm_output = llm.invoke(prompt).strip()
        print("\n🔹 Raw LLM output:", llm_output)

        # Remove unwanted code fences or markdown
        llm_output = llm_output.replace("```json", "").replace("```", "").strip()

        # Validate and repair JSON
        parsed = validate_and_repair_json(llm_output)

        summary = parsed.get("summary", "No summary generated.")
        validated_output = parsed.get("validated_output", structured_output)
        elasticsearch_query = parsed.get("elasticsearch_query", {})

    except Exception as e:
        print(f"❌ Error validating or generating query: {e}")
        summary = f"Error validating or generating query: {e}"
        validated_output = structured_output
        elasticsearch_query = {}

    # ---- Step 5: Return final response ----
    return jsonify({
        "summary": summary,
        "validated_output": structured_output,
        "elasticsearch_query": elasticsearch_query
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)


