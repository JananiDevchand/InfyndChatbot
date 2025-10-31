import os
import json
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

# --------------------------
# Load JSON files as Documents (Optimized for categorical data)
# --------------------------
def load_json_files(data_folder: str):
    """
    Reads all JSON files from the given folder.
    Extracts clean value strings and converts each into a LangChain Document.
    Example:
      {"company_type": "Private Limited Company"} 
      → Document(page_content="Private Limited Company", metadata={"source": "company_type.json", "key": "company_type"})
    """
    documents = []
    for file_name in os.listdir(data_folder):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(data_folder, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Each file can contain a list or a dict
            if isinstance(json_data, list):
                for i, record in enumerate(json_data):
                    if isinstance(record, dict):
                        for key, value in record.items():
                            if isinstance(value, str) and value.strip():
                                documents.append(Document(
                                    page_content=value.strip(),
                                    metadata={"source": file_name, "key": key, "index": i}
                                ))
                    elif isinstance(record, str) and record.strip():
                        documents.append(Document(
                            page_content=record.strip(),
                            metadata={"source": file_name, "index": i}
                        ))

            elif isinstance(json_data, dict):
                for key, value in json_data.items():
                    if isinstance(value, str) and value.strip():
                        documents.append(Document(
                            page_content=value.strip(),
                            metadata={"source": file_name, "key": key}
                        ))

        except Exception as e:
            print(f"❌ Error reading {file_name}: {e}")

    print(f"✅ Loaded {len(documents)} clean documents from {data_folder}")
    return documents


# --------------------------
# Skip text splitting — not needed for short categorical values
# --------------------------
def text_split(extracted_data):
    """
    For categorical short texts, chunking is unnecessary.
    """
    print(f"✅ Using {len(extracted_data)} documents (no chunking required).")
    return extracted_data


# --------------------------
# Load HuggingFace Embedding Model
# --------------------------
def download_hugging_face_embeddings():
    """
    Loads a 384-dimensional embedding model suitable for semantic search.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    print("✅ Loaded HuggingFace embeddings model (384-dim).")
    return embeddings
