import os
from tqdm import tqdm
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from chunker import JSONChunkSplitter
import json

# Paths
PREPROCESSED_JSON = os.path.join("Preprocessed_Json_Files", "preprocessed-2024.json")  # Change as needed
VECTOR_DB_PATH = "collection"
EMBED_MODEL = "nomic-embed-text:v1.5"

# Load new data
with open(PREPROCESSED_JSON, "r", encoding="utf-8") as f:
    new_data = json.load(f)

splitter = JSONChunkSplitter()
documents = splitter.create_documents_from_json(new_data)
print(f"Loaded {len(documents)} new documents.")

# Load embedding model
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# Load existing FAISS DB
if os.path.exists(VECTOR_DB_PATH):
    print("Loading existing FAISS vectorstore...")
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("No existing vectorstore found. Creating new one.")
    db = None

# Embed new documents in batches and merge
BATCH_SIZE = 64
for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Embedding batches"):
    batch_docs = documents[i:i+BATCH_SIZE]
    batch_db = FAISS.from_documents(batch_docs, embeddings)
    if db is None:
        db = batch_db
    else:
        db.merge_from(batch_db)

# Save merged DB
print(f"Saving merged vectorstore to '{VECTOR_DB_PATH}'...")
db.save_local(VECTOR_DB_PATH)
print("Merge complete.")