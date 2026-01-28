"""
MongoDB Knowledge Base Loader

Loads documents from MongoDB for use in the RAG system.
Expected schema:
- _id: MongoDB ObjectId
- title: str
- body: str  
- created_utc: int (Unix timestamp)
- url: str
"""

from typing import List, Dict, Optional
from datetime import datetime, timezone
from loguru import logger

try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    logger.warning("pymongo not available. Install with: pip install pymongo")


class MongoKBLoader:
    """Load knowledge base from MongoDB"""
    
    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017/",
        database: str = "crypto_kb",
        collection: str = "posts"
    ):
        """
        Initialize MongoDB connection.
        
        Args:
            mongo_uri: MongoDB connection string
            database: Database name
            collection: Collection name
        """
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required. Install with: pip install pymongo")
        
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database]
        self.collection = self.db[collection]
        
        logger.info(f"Connected to MongoDB: {database}.{collection}")
    
    def load_documents(
        self,
        limit: Optional[int] = None,
        query: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Load documents from MongoDB.
        
        Args:
            limit: Maximum number of documents to load (None = all)
            query: MongoDB query filter (None = all documents)
            
        Returns:
            List of document dictionaries with fields:
            - id: str (MongoDB _id converted to string)
            - text: str (title + body combined)
            - timestamp: datetime
            - metadata: dict (url, title, source)
        """
        query = query or {}
        
        cursor = self.collection.find(query)
        if limit:
            cursor = cursor.limit(limit)
        
        documents = []
        for doc in cursor:
            # Combine title and body for full text
            title = doc.get('title', '').strip()
            body = doc.get('body', '').strip()
            
            # Combine with clear separation
            if title and body:
                text = f"{title}\n\n{body}"
            elif title:
                text = title
            else:
                text = body
            
            # Skip empty documents
            if not text or len(text) < 10:
                continue
            
            # Parse timestamp
            created_utc = doc.get('created_utc')
            if created_utc:
                timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            
            documents.append({
                'id': str(doc['_id']),
                'text': text,
                'timestamp': timestamp,
                'metadata': {
                    'url': doc.get('url', ''),
                    'title': title,
                    'source': 'mongodb'
                }
            })
        
        logger.info(f"Loaded {len(documents)} documents from MongoDB")
        return documents
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        logger.info("MongoDB connection closed")


if __name__ == "__main__":
    # Example usage
    loader = MongoKBLoader()
    docs = loader.load_documents(limit=10)
    
    print(f"Loaded {len(docs)} documents")
    if docs:
        print("\nFirst document:")
        print(f"ID: {docs[0]['id']}")
        print(f"Text preview: {docs[0]['text'][:200]}...")
        print(f"Timestamp: {docs[0]['timestamp']}")
        print(f"URL: {docs[0]['metadata']['url']}")
    
    loader.close()
