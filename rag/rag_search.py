from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Data class for structured search results
@dataclass
class ProductResult:
    id: str
    title: str
    score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        base.update(
            {
                "price": self.metadata.get("price"),
                "brand": self.metadata.get("brand"),
                "category": self.metadata.get("category"),
                "rating": self.metadata.get("rating"),
                "ingredients": self.metadata.get("ingredients"),
                "product_url": self.metadata.get("product_url"),
                "doc_id": self.id,
            }
        )
        return base

# RAG wrapper around the Chroma collection
class GamesRAG:
    def __init__(
        self,
        vector_dir: Optional[str] = None,
        collection_name: str = "games_accessories",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        if vector_dir is None:
            root_dir = Path(__file__).resolve().parents[1]
            vector_dir = str(root_dir / "data" / "vector_store")

        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(
            path=vector_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(name=collection_name)

    @staticmethod
    def _build_where_filter(
        budget: Optional[float] = None,
        min_price: Optional[float] = None,
        brand: Optional[str] = None,
        category_contains: Optional[str] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        clauses: List[Dict[str, Any]] = []

        if budget is not None and min_price is not None:
            clauses.append({"price": {"$gte": min_price, "$lte": budget}})
        elif budget is not None:
            clauses.append({"price": {"$lte": budget}})
        elif min_price is not None:
            clauses.append({"price": {"$gte": min_price}})

        if brand:
            clauses.append({"brand": {"$eq": brand}})

        if extra_filters:
            for field, condition in extra_filters.items():
                clauses.append({field: condition})

        if not clauses:
            return {}

        if len(clauses) == 1:
            return clauses[0]

        return {"$and": clauses}

    def rag_search(
        self,
        query: str,
        top_k: int = 5,
        budget: Optional[float] = None,
        min_price: Optional[float] = None,
        brand: Optional[str] = None,
        category_contains: Optional[str] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ProductResult]:
        # 1. Encode query
        q_emb = self.model.encode([query]).tolist()

        # 2. Build metadata filter
        where = self._build_where_filter(
            budget=budget,
            min_price=min_price,
            brand=brand,
            category_contains=category_contains,
            extra_filters=extra_filters,
        )

        # 3. Query Chroma
        n_results = max(top_k * 3, top_k)
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=n_results,
            where=where if where else None,
            include=["metadatas", "distances"],
        )

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        out: List[ProductResult] = []
        for pid, meta, dist in zip(ids, metadatas, distances):
            title = ""
            if isinstance(meta, dict):
                title = str(meta.get("title", ""))
            score = 1.0 - float(dist)  # cosine distance → similarity
            out.append(
                ProductResult(
                    id=str(pid),
                    title=title,
                    score=score,
                    metadata=meta,
                )
            )

        if category_contains:
            needle = category_contains.lower()
            out = [
                r
                for r in out
                if needle in str(r.metadata.get("category", "")).lower()
            ]

        # 5. Sort by similarity and trim to top_k
        out.sort(key=lambda r: r.score, reverse=True)
        return out[:top_k]

# Convenience function to import: rag_search(query, filters)
def rag_search(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> List[ProductResult]:
    filters = filters or {}
    rag = GamesRAG()

    return rag.rag_search(
        query=query,
        top_k=top_k,
        budget=filters.get("budget"),
        min_price=filters.get("min_price"),
        brand=filters.get("brand"),
        category_contains=filters.get("category_contains"),
        extra_filters=filters.get("extra_filters"),
    )

# Simple manual test
if __name__ == "__main__":
    example_query = "cooperative family board game for 4 players under $30"
    example_filters = {"budget": 30.0, "category_contains": "Board Games"}

    print(f"Query: {example_query}")
    results = rag_search(example_query, filters=example_filters, top_k=5)

    for i, r in enumerate(results, start=1):
        d = r.to_dict()
        print(f"{i}. {d['title'][:100]}...")
        print(
            f"   id={d['id']}, price={d['price']}, brand={d['brand']}, "
            f"url={d['product_url']}"
        )
        print(f"   score={d['score']:.3f}")