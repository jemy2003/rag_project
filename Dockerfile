FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libsndfile1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement requirements.txt pour tirer parti du cache Docker
COPY requirements.txt .

# Installer les packages Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier seulement le code source
COPY api.py .
COPY .env .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
