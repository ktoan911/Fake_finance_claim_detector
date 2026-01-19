"""
Generic MongoDB Loader for Knowledge Sources

Supports two sources:
  1) Trusted news/official sources
  2) Social media (Reddit/Telegram/etc.)

Only prepares documents d_i for retrieval.
No scoring changes.
"""

from datetime import datetime, timezone
from typing import List, Dict, Optional
from loguru import logger

try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    logger.warning("pymongo not available. Install to load MongoDB sources.")


def _parse_timestamp(value):
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return datetime.now(timezone.utc)


class MongoSourceLoader:
    """
    Load documents from MongoDB with configurable field mapping.
    """

    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required to load from MongoDB.")
        self.client = MongoClient(mongo_uri)
        self.collection = self.client[db_name][collection_name]

    def fetch_documents(
        self,
        text_fields: List[str],
        timestamp_field: str,
        source_label: str,
        link_field: Optional[str] = None,
        limit: int = 5000,
        query: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Returns list of documents compatible with KnowledgeAugmentedRetriever:
          { id, text, timestamp, source, metadata }
        """
        query = query or {}
        cursor = self.collection.find(query).limit(limit)

        documents = []
        for doc in cursor:
            parts = []
            for f in text_fields:
                val = doc.get(f)
                if val:
                    parts.append(str(val).strip())
            text = "\n".join([p for p in parts if p])
            if not text:
                continue

            ts = _parse_timestamp(doc.get(timestamp_field))
            metadata = {"source": source_label}
            if link_field and doc.get(link_field):
                metadata["link"] = doc.get(link_field)

            documents.append({
                "id": str(doc.get("_id")),
                "text": text,
                "timestamp": ts,
                "source": source_label,
                "metadata": metadata,
            })

        logger.info(f"Loaded {len(documents)} documents from {source_label}")
        return documents
