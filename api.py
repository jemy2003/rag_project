import os
import io
import uuid
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
import requests
import PyPDF2

# -------------------------------
# Load environment variables
# -------------------------------
ES_HOST = os.getenv("ES_HOST", "elasticsearch")
ES_PORT = int(os.getenv("ES_PORT", 9200))
ES_INDEX = os.getenv("ES_INDEX", "rag_index")
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

# -------------------------------
# Elasticsearch client
# -------------------------------
es_url = f"http://{ES_HOST}:{ES_PORT}"
es = Elasticsearch(hosts=[es_url], verify_certs=False)

# -------------------------------
# Embedding model
# -------------------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = embed_model.get_sentence_embedding_dimension()

app = FastAPI(title="RAG API with Elasticsearch + Mistral")

# -------------------------------
# Configs
# -------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 5

# -------------------------------
# Ensure index exists (ES8+ compatible)
# -------------------------------
def ensure_index():
    try:
        if not es.indices.exists(index=ES_INDEX):
            mapping = {
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "filename": {"type": "keyword"},
                        "chunk_id": {"type": "integer"},
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": EMBED_DIM,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }
            es.indices.create(index=ES_INDEX, body=mapping)
            print(f"✅ Created index {ES_INDEX} with dims {EMBED_DIM}")
        else:
            print(f"✅ Index '{ES_INDEX}' already exists")
    except Exception as e:
        print(f"❌ Elasticsearch index error: {e}")

ensure_index()

# -------------------------------
# Helpers
# -------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text: str):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP if end - CHUNK_OVERLAP > start else end
    return chunks

def embed_texts(texts: List[str]):
    return embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

def index_document(filename: str, chunks: List[str]):
    doc_id = str(uuid.uuid4())
    embeddings = embed_texts(chunks)
    actions = [
        {
            "_index": ES_INDEX,
            "_id": f"{doc_id}_{idx}",
            "_source": {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_id": idx,
                "text": chunk,
                "embedding": emb
            }
        } for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    helpers.bulk(es, actions)
    return doc_id, len(chunks)

def call_mistral(system_prompt: str, user_prompt: str):
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not configured")
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }
    resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Mistral error: {resp.text}")
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")

def search_es_by_vector(query_vector, top_k=TOP_K):
    script_query = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }
    resp = es.search(index=ES_INDEX, body=script_query)
    hits = resp.get("hits", {}).get("hits", [])
    return [
        {"chunk_id": h["_source"]["chunk_id"], "text": h["_source"]["text"],
         "filename": h["_source"].get("filename"), "score": h.get("_score", 0)}
        for h in hits
    ]

# -------------------------------
# Pydantic models
# -------------------------------
class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K

class UploadResponse(BaseModel):
    doc_id: str
    chunks_indexed: int

# -------------------------------
# Endpoints
# -------------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    text = extract_text_from_pdf(content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")
    chunks = chunk_text(text)
    doc_id, n = index_document(file.filename, chunks)
    return {"doc_id": doc_id, "chunks_indexed": n}

@app.post("/ask")
def ask(request: AskRequest):
    q_emb = embed_model.encode([request.question], convert_to_numpy=True)[0].tolist()
    hits = search_es_by_vector(q_emb, top_k=request.top_k)
    if not hits:
        return {"answer": "No relevant passages found.", "sources": [], "hits": 0}
    joined_context = "\n\n---\n\n".join([h["text"] for h in hits])
    system_prompt = "You are a helpful assistant. Use the provided context to answer the user's question."
    answer = call_mistral(system_prompt, f"Context:\n{joined_context}\n\nQuestion: {request.question}")
    return {"answer": answer, "sources": hits, "hits": len(hits)}
