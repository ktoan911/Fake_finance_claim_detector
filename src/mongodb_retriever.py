import os
from typing import List

from loguru import logger

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection

    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    logger.warning("pymongo not installed. MongoDBRetriever will not work.")

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("sentence_transformers not installed.")


class MongoDBRetriever:
    """
    Retrieves evidence from MongoDB using vector search or text search.
    Assumes documents in MongoDB have 'text' and 'embedding' fields.
    """

    def __init__(
        self,
        uri: str = None,
        db_name: str = "crypto_claims",
        collection_name: str = "evidence",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        vector_index_name: str = "vector_index",
    ):
        if not MONGO_AVAILABLE:
            raise ImportError("pymongo is required.")

        self.uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.client = MongoClient(self.uri)
        self.db = self.client[db_name]
        self.collection: Collection = self.db[collection_name]
        self.vector_index_name = vector_index_name

        if EMBEDDING_AVAILABLE:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.encoder = SentenceTransformer(embedding_model)
        else:
            self.encoder = None

        logger.info(
            f"MongoDBRetriever initialized. DB: {db_name}, Coll: {collection_name}"
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve relevant evidence for a query.
        Uses Vector Search if supported (Atlas Search), otherwise falls back to basic text matching
        (or you can implement manual cosine similarity if dataset is small).

        For this implementation, we will assume Atlas Vector Search or similar if 'embedding' exists.
        If not, we'll do a simple text find (regex) as a fallback (NOT RECOMMENDED for production).
        """
        if not query:
            return []

        # 1. Generate Query Embedding
        query_embedding = None
        if self.encoder:
            query_embedding = self.encoder.encode(query, convert_to_numpy=True).tolist()

        results = []

        # 2. Try Atlas Vector Search (Aggregation Pipeline)
        # Note: This requires an Atlas Search Index defined on the collection.
        if query_embedding:
            try:
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": self.vector_index_name,
                            "path": "embedding",
                            "queryVector": query_embedding,
                            "numCandidates": top_k * 10,
                            "limit": top_k,
                        }
                    },
                    {"$project": {"text": 1, "score": {"$meta": "vectorSearchScore"}}},
                ]
                cursor = self.collection.aggregate(pipeline)
                results = list(cursor)
                logger.info(f"Atlas Vector Search returned {len(results)} results.")
            except Exception as e:
                logger.warning(
                    f"Atlas Vector Search failed (Index might be missing or not on Atlas): {e}"
                )
                results = []

        # 3. Fallback: If Vector Search failed or returned nothing, try simple text search
        # Or if we want to do Client-Side Re-ranking (only for small datasets)
        if not results:
            logger.info("Falling back to standard text search (regex/text index)...")
            # Simple text search (requires text index or regex)
            # Using regex for flexibility in this demo (slow for large DBs)
            cursor = self.collection.find(
                {"text": {"$regex": query, "$options": "i"}}, {"text": 1, "_id": 0}
            ).limit(top_k)
            results = list(cursor)

        # Extract text
        evidence_texts = [doc.get("text", "") for doc in results if doc.get("text")]
        return evidence_texts

    def close(self):
        self.client.close()
