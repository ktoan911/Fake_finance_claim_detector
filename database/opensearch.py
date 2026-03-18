from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from dotenv import load_dotenv
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

load_dotenv()

JsonDict = Dict[str, Any]


@dataclass
class SearchHit:
    id: str
    score: float
    source: JsonDict


class OpenSearchKB:
    def __init__(
        self,
        host: str = os.getenv("OP_HOST"),
        port: int = int(os.getenv("OP_PORT")),
        auth: Tuple[str, str] = (
            os.getenv("OP_AUTH_USERNAME"),
            os.getenv("OP_AUTH_PASSWORD"),
        ),
        index_name: str = os.getenv("OP_KB_NAME"),
        embedding_dim: int = int(os.getenv("OP_EMBEDDING_DIM")),
        id_field_candidates: Sequence[str] = ("_id", "id"),
    ):
        self.index = index_name
        self.embedding_dim = embedding_dim
        self.id_field_candidates = tuple(id_field_candidates)

        self.client = OpenSearch(
            hosts=[{"host": host, "port": port, "scheme": "https"}],
            http_auth=auth,
            verify_certs=True,
            http_compress=True,
            timeout=60,
            max_retries=3,
            retry_on_timeout=True,
        )

    def create_index(self, overwrite: bool = False) -> None:
        """
        Create index with mappings for BM25 + k-NN vector.
        If overwrite=True, delete index first if exists.
        """
        exists = self.client.indices.exists(index=self.index)
        if exists and overwrite:
            self.client.indices.delete(index=self.index)
            exists = False

        if exists:
            return

        body = {
            "settings": {
                "index": {
                    "knn": True,
                    # you can tune these later
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "description": {"type": "text"},
                    "content": {"type": "text"},
                    "timestamp": {"type": "text"},  # ISO8601 recommended
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        # optional: method tuning (HNSW)
                        # "method": {
                        #   "name": "hnsw",
                        #   "engine": "nmslib",
                        #   "space_type": "cosinesimil",
                        #   "parameters": {"ef_construction": 128, "m": 16}
                        # }
                    },
                }
            },
        }

        self.client.indices.create(index=self.index, body=body)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _extract_id(self, doc: JsonDict) -> Optional[str]:
        for k in self.id_field_candidates:
            v = doc.get(k)
            if v is not None and str(v).strip() != "":
                return str(v)
        return None

    def _strip_id_fields(self, doc: JsonDict) -> JsonDict:
        d = dict(doc)
        for k in self.id_field_candidates:
            d.pop(k, None)
        return d

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", str(query or "")).strip()

    @staticmethod
    def _split_field_boost(field_name: str) -> Tuple[str, float]:
        raw = str(field_name or "").strip()
        if not raw:
            return "", 1.0
        if "^" not in raw:
            return raw, 1.0
        name, boost_raw = raw.split("^", 1)
        try:
            boost = float(boost_raw)
        except ValueError:
            boost = 1.0
        return name.strip(), boost

    # ----------------------------
    # Write operations
    # ----------------------------
    def insert_many(
        self,
        docs: List[JsonDict],
        refresh: Union[bool, str] = "wait_for",
        upsert: bool = True,
        chunk_size: int = 500,
        raise_on_error: bool = True,
    ) -> JsonDict:
        """
        Bulk insert/upsert list of dicts.

        If upsert=True:
          - uses "index" (overwrite by _id)
        If upsert=False:
          - uses "create" (fail if exists)
        """
        if not self.client.indices.exists(index=self.index):
            self.create_index(overwrite=False)
        if not docs:
            return {"inserted": 0, "errors": 0}

        actions = []
        op_type = "index" if upsert else "create"

        for doc in docs:
            doc_id = self._extract_id(doc)
            if doc_id is None:
                raise ValueError(
                    "Each doc must include an id field ('_id' or 'id') to be safely managed. "
                    "Add it before insert."
                )
            source = self._strip_id_fields(doc)
            actions.append(
                {
                    "_op_type": op_type,
                    "_index": self.index,
                    "_id": doc_id,
                    "_source": source,
                }
            )

        success, errors = bulk(
            self.client,
            actions,
            chunk_size=chunk_size,
            raise_on_error=raise_on_error,
            refresh=refresh,
        )

        return {
            "inserted": int(success),
            "errors": len(errors) if errors else 0,
            "error_items": errors or [],
        }

    def delete_many(
        self,
        items: List[Union[str, JsonDict]],
        refresh: Union[bool, str] = "wait_for",
        chunk_size: int = 500,
        raise_on_error: bool = True,
    ) -> JsonDict:
        """
        Bulk delete by ids.

        items can be:
          - list[str] (ids)
          - list[dict] where dict contains "_id" or "id"
        """
        if not items:
            return {"deleted": 0, "errors": 0}

        actions = []
        for it in items:
            if isinstance(it, str):
                doc_id = it
            elif isinstance(it, dict):
                doc_id = self._extract_id(it)
            else:
                raise TypeError("items must be list[str] or list[dict].")

            if doc_id is None:
                raise ValueError(
                    "Delete items must have an id (string or dict with '_id'/'id')."
                )

            actions.append(
                {
                    "_op_type": "delete",
                    "_index": self.index,
                    "_id": str(doc_id),
                }
            )

        success, errors = bulk(
            self.client,
            actions,
            chunk_size=chunk_size,
            raise_on_error=raise_on_error,
            refresh=refresh,
        )

        return {
            "deleted": int(success),
            "errors": len(errors) if errors else 0,
            "error_items": errors or [],
        }

    # ----------------------------
    # Read operations (Search)
    # ----------------------------
    def search_bm25(
        self,
        query: str,
        k: int = 10,
        fields: Optional[List[str]] = None,
        filters: Optional[JsonDict] = None,
        min_timestamp: Optional[str] = None,
        max_timestamp: Optional[str] = None,
    ) -> List[SearchHit]:
        """
        BM25 search over text fields.
        - filters: OpenSearch bool filter clauses (term/range/etc)
        - min_timestamp/max_timestamp: ISO date strings
        """
        if fields is None:
            fields = ["title^3", "description^2", "content"]

        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return []

        tokens = re.findall(r"\w+", normalized_query)
        token_count = len(tokens)
        phrase_query = (
            normalized_query if token_count <= 32 else " ".join(tokens[:32]).strip()
        )

        should_queries: List[JsonDict] = [
            {
                "multi_match": {
                    "query": normalized_query,
                    "fields": fields,
                    "type": "best_fields",
                    "boost": 1.0,
                }
            }
        ]

        if token_count >= 3:
            should_queries.append(
                {
                    "multi_match": {
                        "query": normalized_query,
                        "fields": fields,
                        "type": "cross_fields",
                        "operator": "and",
                        "boost": 2.0,
                    }
                }
            )

        if token_count >= 6:
            minimum_should_match = "60%" if token_count < 12 else "70%"
            should_queries.append(
                {
                    "multi_match": {
                        "query": normalized_query,
                        "fields": fields,
                        "type": "most_fields",
                        "minimum_should_match": minimum_should_match,
                        "boost": 1.5,
                    }
                }
            )

        if phrase_query and token_count >= 4:
            for raw_field in fields:
                base_field, boost = self._split_field_boost(raw_field)
                if not base_field:
                    continue
                should_queries.append(
                    {
                        "match_phrase": {
                            base_field: {
                                "query": phrase_query,
                                "slop": 2,
                                "boost": max(2.0, boost * 4.0),
                            }
                        }
                    }
                )

        filter_clauses = []
        if filters:
            # Expect filters already in OpenSearch DSL forms or simple terms.
            # If you pass {"term": {"some_field": "x"}} it will be appended directly.
            # If you pass {"some_field": "x"} it will be converted to term.
            if "term" in filters or "range" in filters or "bool" in filters:
                filter_clauses.append(filters)
            else:
                for fk, fv in filters.items():
                    filter_clauses.append({"term": {fk: fv}})

        if min_timestamp or max_timestamp:
            range_body: JsonDict = {}
            if min_timestamp:
                range_body["gte"] = min_timestamp
            if max_timestamp:
                range_body["lte"] = max_timestamp
            filter_clauses.append({"range": {"timestamp": range_body}})

        body = {
            "size": k,
            "query": {
                "bool": {
                    "should": should_queries,
                    "minimum_should_match": 1,
                    "filter": filter_clauses if filter_clauses else [],
                }
            },
        }

        resp = self.client.search(index=self.index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [
            SearchHit(
                id=str(h.get("_id")),
                score=float(h.get("_score", 0.0) or 0.0),
                source=h.get("_source", {}) or {},
            )
            for h in hits
        ]

    def search_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        filters: Optional[JsonDict] = None,
        min_timestamp: Optional[str] = None,
        max_timestamp: Optional[str] = None,
    ) -> List[SearchHit]:
        """
        Vector k-NN search on 'embedding'.

        Note:
        - Some OpenSearch versions accept: {"query": {"knn": {...}}}
        - Others support newer "knn" query styles.
        This implementation uses the common pattern.
        """
        if len(query_vector) != self.embedding_dim:
            raise ValueError(
                f"query_vector dim mismatch: got {len(query_vector)} expected {self.embedding_dim}"
            )

        # kNN part
        num_candidates = max(k * 8, 200)
        knn_query: JsonDict = {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                    "num_candidates": num_candidates,
                }
            }
        }
        # filters
        filter_clauses = []
        if filters:
            if "term" in filters or "range" in filters or "bool" in filters:
                filter_clauses.append(filters)
            else:
                for fk, fv in filters.items():
                    filter_clauses.append({"term": {fk: fv}})

        if min_timestamp or max_timestamp:
            range_body: JsonDict = {}
            if min_timestamp:
                range_body["gte"] = min_timestamp
            if max_timestamp:
                range_body["lte"] = max_timestamp
            filter_clauses.append({"range": {"timestamp": range_body}})

        # If you need filtering with knn, typical approach is to wrap in bool.
        # Some OpenSearch versions require knn at top-level query; others allow bool.
        # We'll use bool wrapping (works in many DO-managed OpenSearch setups).
        body: JsonDict = {
            "size": k,
            "query": {
                "bool": {
                    "must": [knn_query],
                    "filter": filter_clauses if filter_clauses else [],
                }
            },
        }

        from opensearchpy.exceptions import RequestError

        try:
            resp = self.client.search(index=self.index, body=body)
        except RequestError as exc:
            # Older OpenSearch versions may not support `num_candidates`.
            if "num_candidates" in str(exc).lower():
                knn_query["knn"]["embedding"].pop("num_candidates", None)
                resp = self.client.search(index=self.index, body=body)
            else:
                raise
        except Exception as exc:
            if "num_candidates" in str(exc).lower():
                knn_query["knn"]["embedding"].pop("num_candidates", None)
                resp = self.client.search(index=self.index, body=body)
            else:
                raise

        hits = resp.get("hits", {}).get("hits", [])
        return [
            SearchHit(
                id=str(h.get("_id")),
                score=float(h.get("_score", 0.0) or 0.0),
                source=h.get("_source", {}) or {},
            )
            for h in hits
        ]


# if __name__ == "__main__":

#     kb = OpenSearchKB(index_name="kb-news", embedding_dim=768)

#     # Create index once
#     kb.create_index(overwrite=False)

#     # Insert docs
#     docs = [
#         {
#             "id": "post-1",
#             "title": "Hello",
#             "description": "short desc",
#             "timestamp": "12345",
#             "content": "this is a document about solar panel snow",
#             "embedding": [0.0] * 768,
#         }
#     ]
#     print(kb.insert_many(docs))

#     # BM25 search
#     hits_bm25 = kb.search_bm25("solar panel snow", k=5)
#     print("BM25:", [(h.id, h.score) for h in hits_bm25])

#     # Vector search
#     hits_vec = kb.search_vector([0.0] * 768, k=5)
#     print("VEC:", [(h.id, h.score) for h in hits_vec])

#     # Delete docs by id
#     # print(kb.delete_many(["post-1"]))
#     print("indices:", kb.client.cat.indices(index="kb-news", format="json"))
#     print("count:", kb.client.count(index="kb-news"))
#     print("get:", kb.client.get(index="kb-news", id="post-1"))
