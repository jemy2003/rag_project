# 🤖 RAG Assistant — FastAPI + Elasticsearch + Mistral AI

Un assistant RAG complet permettant d’ingérer des PDF, de stocker leurs embeddings dans Elasticsearch, puis de répondre à des questions grâce à Mistral AI.  
Simple, rapide, efficace.  

---

## 🚀 Fonctionnalités

- **Upload de PDF**  
  Extraction automatique du texte + segmentation en chunks.

- **Embeddings (all-MiniLM-L6-v2)**  
  Vectorisation hautes-performances via sentence-transformers.

- **Stockage sémantique (Elasticsearch 8+)**  
  Utilise `dense_vector` + scoring `cosineSimilarity()`.

- **RAG complet**  
  1. Recherche vectorielle  
  2. Sélection des passages les plus pertinents  
  3. Réponse finale générée via Mistral AI

- **FastAPI**  
  - `POST /upload` → ingère un PDF  
  - `POST /ask` → exécute une requête RAG  

---

## 🧱 Architecture du projet

rag_project/
├── api.py # API FastAPI principale

├── Dockerfile # Image Docker de l’API

├── docker-compose.yml # Lance Elasticsearch + API

├── wait_for_es.sh # Script pour attendre Elasticsearch avant démarrage

├── requirements.txt # Dépendances Python

├── .env # Variables d’environnement

└── uploads/

  └── Introduction to Data Engineering.pdf
