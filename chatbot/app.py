from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from src.prompt import *
from pymongo import MongoClient
import os
import json
from datetime import datetime

# ------------------- Flask App -------------------
app = Flask(__name__)

# ------------------- Load Environment Variables -------------------
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')  # fallback to local

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ------------------- MongoDB Setup -------------------
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["chatbot_dbbbb"]
    chat_collection = db["chat_history"]
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print("❌ MongoDB connection error:", e)

# ------------------- Embeddings and Vector Store -------------------
embeddings = download_hugging_face_embeddings()
index_name = "infyndcompanydata"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ------------------- LLM Model -------------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# ------------------- Retrieval Chain -------------------
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ------------------- Routes -------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    print(f"🧠 User Input: {msg}")

    # Invoke RAG model
    response = rag_chain.invoke({"input": msg})
    raw_answer = response.get("answer", "")

    # ------------------- Parse JSON output -------------------
    if "{" in raw_answer:
        parts = raw_answer.split("{", 1)
        human_part = parts[0].strip()
        json_part = "{" + parts[1]
    else:
        human_part = raw_answer
        json_part = "{}"

    try:
        parsed_json = json.loads(json_part)
        filters = parsed_json.get("filters", {})
    except json.JSONDecodeError:
        filters = {}

    # ------------------- Prepare Final Output -------------------
    final_output = {
        "user_input": msg,
        "answer": human_part,
        "filters": filters,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # ------------------- Save to MongoDB -------------------
    try:
        result = chat_collection.insert_one(final_output)
        final_output["_id"] = str(result.inserted_id)  # convert ObjectId for JSON serialization
        print("💾 Chat saved to MongoDB")
    except Exception as e:
        print("⚠️ Failed to save chat:", e)

    print("✅ Response:", final_output)
    return jsonify(final_output)


@app.route("/history", methods=["GET"])
def history():
    """Retrieve last 10 chat entries"""
    try:
        chats = list(chat_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(10))
        return jsonify(chats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------- Run Flask App -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
