#!/usr/bin/env python3
"""
Example: Verify a claim using the Fusion model + MongoDB KB
"""

from src.mongo_loader import MongoKBLoader
import sys

# Quick test to see MongoDB connection works
try:
    loader = MongoKBLoader(
        mongo_uri="mongodb://localhost:27017/",
        database="crypto_kb",
        collection="posts"
    )
    
    docs = loader.load_documents(limit=5)
    
    print(f"✓ MongoDB connected successfully!")
    print(f"✓ Loaded {len(docs)} sample documents")
    
    if docs:
        print(f"\nSample document:")
        print(f"  ID: {docs[0]['id']}")
        print(f"  Title: {docs[0]['metadata']['title'][:50]}...")
        print(f"  Text length: {len(docs[0]['text'])} chars")
        print(f"  Timestamp: {docs[0]['timestamp']}")
    
    loader.close()
    
    print("\n" + "="*60)
    print("To run inference, use:")
    print('  python inference_fusion.py --claim "Your claim here"')
    print("="*60)
    
except Exception as e:
    print(f"✗ Error connecting to MongoDB: {e}")
    print("\nMake sure:")
    print("  1. MongoDB is running")
    print("  2. Database 'crypto_kb' exists")
    print("  3. Collection 'posts' has documents")
    print("  4. pymongo is installed: pip install pymongo")
    sys.exit(1)
