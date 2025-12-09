import math
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

DATA_DIR = Path("data")
PROCESSED_PATH = DATA_DIR / "processed" / "games_accessories.parquet"
VECTOR_DIR = DATA_DIR / "vector_store"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the processed subset
df = pd.read_parquet(PROCESSED_PATH)
print("Loaded processed subset:", df.shape)

# 2. Build the text used for embeddings: title + features
def build_text(row) -> str:
    parts = [
        str(row.get("title", "")),
        str(row.get("features", "")),
    ]
    parts = [p.strip() for p in parts if p and p.strip()]
    return " | ".join(parts)

df["text_for_embedding"] = df.apply(build_text, axis=1)

# 3. Choose the embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# 4. Initialize Chroma (persistent vector store)
client = chromadb.PersistentClient(
    path=str(VECTOR_DIR),
    settings=Settings(anonymized_telemetry=False),
)

collection_name = "games_accessories"

# Re-create the collection to avoid duplicates when re-running
try:
    client.delete_collection(name=collection_name)
except Exception:
    pass

collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"},
)

# 5. Prepare metadata for each row
def row_to_metadata(row):
    meta = {}
    for col in df.columns:
        if col in ["text_for_embedding"]:
            continue
        val = row[col]
        if isinstance(val, (list, dict)):
            val = str(val)
        if pd.isna(val):
            continue
        meta[col] = val
    return meta

ids = df["id"].astype(str).tolist()
texts = df["text_for_embedding"].tolist()
metadatas = [row_to_metadata(row) for _, row in df.iterrows()]

# 6. Batch-encode and write to Chroma
batch_size = 128
num_batches = math.ceil(len(texts) / batch_size)

for i in range(num_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(texts))
    batch_texts = texts[start:end]
    batch_ids = ids[start:end]
    batch_metas = metadatas[start:end]

    embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
    collection.add(
        ids=batch_ids,
        embeddings=embeddings,
        documents=batch_texts,
        metadatas=batch_metas,
    )
    print(f"Inserted batch {i+1}/{num_batches}: {end - start} items")

print("Vector index built successfully.")