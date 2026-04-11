#!/usr/bin/env python3
"""
Example pipeline:
1) Take a list of claim sentences
2) Cluster claims by semantic similarity
3) Call an LLM hook to generate content/description for each cluster
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "sentence-transformers is required. Install dependencies from requirements.txt."
    ) from exc

client = OpenAI(
    api_key="D",
    base_url="https://api-chua.onrender.com/v1",
)


def generate_cluster_content_with_llm(
    cluster_claims: List[str], representative_claim: str
) -> str:

    cluster_all = ""

    _resp = client.chat.completions.create(
        model="gemini-3.0-pro",
        messages=[{"role": "user", "content": cluster_claims}],
    )
    return _resp.choices[0].message.content


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def compute_embeddings(
    claims: List[str], model_name: str, batch_size: int = 32
) -> np.ndarray:
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            claims,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
    except Exception as exc:
        print(
            f"[Warning] Could not load embedding model '{model_name}'. "
            f"Fallback to TF-IDF embeddings. Error: {exc}"
        )
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        tfidf = vectorizer.fit_transform(claims)
        dense = tfidf.toarray().astype(np.float32)
        return _normalize_rows(dense)


def choose_cluster_count(
    embeddings: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> int:
    n_samples = embeddings.shape[0]
    if n_samples <= 2:
        return 1

    lower = max(2, min_k)
    upper = min(max_k, n_samples - 1)
    if lower > upper:
        return 1

    best_k = 1
    best_score = -1.0
    for k in range(lower, upper + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _find_representative_claim(
    cluster_embeddings: np.ndarray, cluster_claims: List[str]
) -> str:
    centroid = np.mean(cluster_embeddings, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return cluster_claims[0]

    centroid = centroid / centroid_norm
    normalized = _normalize_rows(cluster_embeddings)
    similarities = normalized @ centroid
    rep_idx = int(np.argmax(similarities))
    return cluster_claims[rep_idx]


def cluster_claims(
    claims: List[str],
    model_name: str,
    num_clusters: Optional[int] = None,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> Dict:
    cleaned_claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
    if not cleaned_claims:
        raise ValueError("Input claims list is empty after cleaning.")

    embeddings = compute_embeddings(cleaned_claims, model_name=model_name)
    n_samples = len(cleaned_claims)

    if num_clusters is None:
        k = choose_cluster_count(
            embeddings=embeddings,
            min_k=min_k,
            max_k=max_k,
            random_state=random_state,
        )
    else:
        if num_clusters < 1:
            raise ValueError("num_clusters must be >= 1")
        k = min(num_clusters, n_samples)

    if k == 1:
        labels = np.zeros(n_samples, dtype=int)
    else:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(embeddings)

    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_to_indices[int(label)].append(idx)

    clusters = []
    for cluster_id in sorted(cluster_to_indices):
        indices = cluster_to_indices[cluster_id]
        cluster_claim_list = [cleaned_claims[i] for i in indices]
        cluster_vectors = embeddings[indices]
        representative_claim = _find_representative_claim(
            cluster_embeddings=cluster_vectors, cluster_claims=cluster_claim_list
        )

        generated_content = generate_cluster_content_with_llm(
            cluster_claims=cluster_claim_list,
            representative_claim=representative_claim,
        )

        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(indices),
                "representative_claim": representative_claim,
                "cluster_content": generated_content,
                "claims": cluster_claim_list,
            }
        )

    return {
        "num_input_claims": n_samples,
        "num_clusters": len(clusters),
        "model_name": model_name,
        "clusters": clusters,
    }


def load_claims_from_file(path: str) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if file_path.suffix.lower() == ".txt":
        return [
            line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()
        ]

    if file_path.suffix.lower() == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(item) for item in data]
        if (
            isinstance(data, dict)
            and "claims" in data
            and isinstance(data["claims"], list)
        ):
            return [str(item) for item in data["claims"]]
        raise ValueError("JSON must be a list[str] or {'claims': list[str]}.")

    raise ValueError("Only .txt or .json claim files are supported.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster claims and summarize each cluster with an LLM hook."
    )
    parser.add_argument(
        "--claims_file",
        type=str,
        default=None,
        help="Path to .txt (one claim per line) or .json (list[str] or {'claims': [...]})",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model name for sentence-transformers.",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=None,
        help="Fixed number of clusters. If omitted, auto-select with silhouette.",
    )
    parser.add_argument(
        "--min_k",
        type=int,
        default=2,
        help="Minimum k when auto-selecting cluster count.",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=10,
        help="Maximum k when auto-selecting cluster count.",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional output JSON path."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.claims_file:
        claims = load_claims_from_file(args.claims_file)
    else:
        claims = [
            "Bitcoin sẽ đạt 200k USD trong 3 tháng tới.",
            "Giá BTC có thể lên 200 nghìn đô trong quý này.",
            "Airdrop token XYZ chắc chắn lời 100%, không có rủi ro.",
            "Dự án XYZ airdrop đảm bảo lợi nhuận tuyệt đối.",
            "ETF spot mới có thể khiến dòng tiền vào thị trường crypto tăng.",
            "Việc phê duyệt ETF spot có thể đẩy thanh khoản thị trường tiền mã hóa.",
            "Mua coin theo tín hiệu nội gián là cách kiếm lời nhanh nhất.",
            "Có nguồn tin nội bộ, mua ngay để x5 tài khoản.",
        ]

    result = cluster_claims(
        claims=claims,
        model_name=args.model_name,
        num_clusters=args.num_clusters,
        min_k=args.min_k,
        max_k=args.max_k,
    )

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"\nSaved output to: {args.output}")


if __name__ == "__main__":
    main()
