from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from loguru import logger

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using numpy-based embeddings only.")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class TripletSample:
    """Container for triplet training sample"""

    anchor: str  # Query
    positive: str  # Matching scam pattern
    negatives: List[str]  # List of non-matching samples (hard + random)


class SimulatedEmbeddingModel:
    """
    Simulated embedding model for testing without heavy dependencies.
    Uses TF-IDF with dimension reduction.

    This is the main embedding model used when PyTorch/SentenceTransformers
    are not available.
    """

    def __init__(self, embedding_dim: int = 128, seed: int = 42):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 2), stop_words="english"
        )
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=seed)
        self.is_fitted = False

    def fit(self, texts: List[str]) -> None:
        """Fit the embedding model"""
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self.is_fitted = True
        logger.info(f"SimulatedEmbeddingModel fitted on {len(texts)} texts")

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # TF-IDF + SVD
        tfidf = self.vectorizer.transform(texts)
        embeddings = self.svd.transform(tfidf)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            embeddings = embeddings / norms

        return embeddings

    def similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """Compute cosine similarity between query and documents"""
        query_emb = self.encode([query])
        doc_embs = self.encode(documents)

        similarities = np.dot(doc_embs, query_emb.T).flatten()
        return similarities


# PyTorch-based classes (only available if torch is installed)
if TORCH_AVAILABLE:

    class CryptoEmbeddingDataset(TorchDataset):
        """
        Dataset for contrastive learning on crypto scam patterns.
        Creates triplets (q, d+, {d-}) for training.
        Supports hard negative mining and multiple negatives.
        """

        def __init__(
            self,
            scam_samples: List[Dict],
            legitimate_samples: List[Dict],
            num_triplets: int = 10000,
            num_negatives: int = 3,  # Number of negatives per anchor
            hard_negative_ratio: float = 0.5,
            embedding_model=None,  # Optional model for hard negative mining
        ):
            self.scam_samples = scam_samples
            self.legitimate_samples = legitimate_samples
            self.num_negatives = num_negatives
            self.hard_negative_ratio = hard_negative_ratio
            self.embedding_model = embedding_model

            # Group scams by type
            self.scams_by_type = {}
            for sample in scam_samples:
                scam_type = sample.get("scam_type", "unknown")
                if scam_type not in self.scams_by_type:
                    self.scams_by_type[scam_type] = []
                self.scams_by_type[scam_type].append(sample)

            # Pre-compute embeddings for hard negative mining if model provided
            self.legit_embeddings = None
            if self.embedding_model and hasattr(self.embedding_model, "encode"):
                logger.info(
                    "Pre-computing legitimate embeddings for hard negative mining..."
                )
                legit_texts = [s["text"] for s in self.legitimate_samples]
                # We use CPU for storage to avoid OOM
                self.legit_embeddings = self.embedding_model.encode(
                    legit_texts, normalize=True
                ).cpu()

            # Generate triplets
            self.triplets = self._generate_triplets(num_triplets)
            logger.info(f"Created dataset with {len(self.triplets)} triplets")

        def _generate_triplets(self, num_triplets: int) -> List[TripletSample]:
            """Generate triplet samples with hard negative mining"""
            triplets = []
            scam_types = list(self.scams_by_type.keys())

            for _ in range(num_triplets):
                anchor_type = np.random.choice(scam_types)
                anchor_samples = self.scams_by_type[anchor_type]

                # We need at least 1 sample.
                # If using Claim-Evidence, 1 is enough.
                # If using Claim-Claim (fallback), we need 2.
                if not anchor_samples:
                    continue

                # Pick anchor
                anchor_idx = np.random.choice(len(anchor_samples))
                anchor_sample = anchor_samples[anchor_idx]
                anchor_text = anchor_sample["text"]

                # Determine Positive
                # Priority: Use "evidence" field (Paper alignment: Query -> Evidence)
                if "evidence" in anchor_sample and anchor_sample["evidence"]:
                    positive_text = anchor_sample["evidence"]
                    using_evidence = True
                elif len(anchor_samples) >= 2:
                    # Fallback: Metric Learning (Cluster same scam types)
                    # Pick another sample of same type
                    pos_idx = np.random.choice(
                        [i for i in range(len(anchor_samples)) if i != anchor_idx]
                    )
                    positive_text = anchor_samples[pos_idx]["text"]
                    using_evidence = False
                else:
                    # Cannot form pair
                    continue

                negatives = []

                # Try to mine hard negatives
                if self.legit_embeddings is not None:
                    # Find legitimate sample that is semantically similar to anchor
                    anchor_emb = self.embedding_model.encode(
                        [anchor_text], normalize=True
                    ).cpu()

                    # Compute cosine similarity
                    sims = F.cosine_similarity(anchor_emb, self.legit_embeddings, dim=1)

                    # Select top-k similar but not identical
                    top_k_indices = torch.topk(
                        sims, k=min(20, len(self.legitimate_samples))
                    ).indices

                    # Add hard negatives
                    num_hard = int(self.num_negatives * self.hard_negative_ratio)
                    for _ in range(num_hard):
                        neg_idx = top_k_indices[
                            np.random.randint(0, len(top_k_indices))
                        ].item()
                        negatives.append(self.legitimate_samples[neg_idx]["text"])

                # Fill remaining with random negatives
                while len(negatives) < self.num_negatives:
                    if np.random.random() < 0.5:
                        # Random negative from other scam types
                        other_types = [t for t in scam_types if t != anchor_type]
                        if other_types:
                            neg_type = np.random.choice(other_types)
                            neg_sample = np.random.choice(self.scams_by_type[neg_type])

                            # If we are doing Claim-Evidence, try to pick Evidence as negative too
                            if (
                                using_evidence
                                and "evidence" in neg_sample
                                and neg_sample["evidence"]
                            ):
                                negatives.append(neg_sample["evidence"])
                            else:
                                negatives.append(neg_sample["text"])
                        else:
                            negatives.append(
                                np.random.choice(self.legitimate_samples)["text"]
                            )
                    else:
                        # Random legitimate sample
                        negatives.append(
                            np.random.choice(self.legitimate_samples)["text"]
                        )

                triplets.append(
                    TripletSample(
                        anchor=anchor_text, positive=positive_text, negatives=negatives
                    )
                )

            return triplets

        def __len__(self) -> int:
            return len(self.triplets)

        def __getitem__(self, idx: int) -> Dict[str, Union[str, List[str]]]:
            triplet = self.triplets[idx]
            return {
                "anchor": triplet.anchor,
                "positive": triplet.positive,
                "negatives": triplet.negatives,
            }

    class ContrastiveEmbeddingModel(nn.Module):
        """
        Embedding model with contrastive fine-tuning capability.
        Implements Equation (7):
        L = -log(e^s+ / (e^s+ + Σe^s-)) + λ||θ||²
        """

        def __init__(
            self,
            base_model_name: str = "BAAI/bge-small-en-v1.5",
            embedding_dim: int = 384,
            lambda_reg: float = 0.001,
            freeze_base: bool = True,
        ):
            super().__init__()

            self.lambda_reg = lambda_reg
            self.embedding_dim = embedding_dim

            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.encoder = SentenceTransformer(base_model_name)
                if freeze_base:
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
            else:
                self.encoder = None

            self.projection = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )

            logger.info(
                f"ContrastiveEmbeddingModel initialized: dim={self.embedding_dim}, λ={lambda_reg}"
            )

        def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
            """Encode texts to embeddings."""
            device = next(self.projection.parameters()).device

            if self.encoder is not None:
                # If we are training and base is not frozen, we need gradients
                # SentenceTransformer encode by default uses no_grad(), so we must bypass it
                # or use its deeper forward methods.
                # However, an easier way to get gradients from HuggingFace models inside SentenceTransformer
                # is to pass tokenized input directly to the core transformer model.

                # As a workaround to sentence-transformers' `.encode()` which is hardcoded for inference:
                if self.training and any(
                    p.requires_grad for p in self.encoder.parameters()
                ):
                    # Tokenize
                    features = self.encoder.tokenize(texts)
                    features = {k: v.to(device) for k, v in features.items()}
                    # Forward pass through base model to get gradients
                    out_features = self.encoder(features)
                    base_embeddings = out_features["sentence_embedding"]
                else:
                    with torch.no_grad():
                        base_embeddings = self.encoder.encode(
                            texts, convert_to_tensor=True, show_progress_bar=False
                        )
                        base_embeddings = base_embeddings.to(device)
            else:
                base_embeddings = torch.randn(len(texts), self.embedding_dim).to(device)

            projected = self.projection(base_embeddings)

            if normalize:
                projected = F.normalize(projected, p=2, dim=-1)

            return projected

        def compute_similarity(
            self, anchor_emb: torch.Tensor, target_emb: torch.Tensor
        ) -> torch.Tensor:
            """Compute cosine similarity between embeddings"""
            return F.cosine_similarity(anchor_emb, target_emb)

        def compute_contrastive_loss(
            self,
            anchor_emb: torch.Tensor,
            positive_emb: torch.Tensor,
            negative_embs: torch.Tensor,  # Can be [B, N_neg, Dim] or [B, Dim]
            temperature: float = 0.07,
        ) -> torch.Tensor:
            """
            Compute contrastive loss (Equation 7) with multiple negatives support:
            L = -log(e^s+ / (e^s+ + Σe^s-)) + λ||θ||²
            """
            # Positive similarity: [B]
            pos_sim = self.compute_similarity(anchor_emb, positive_emb) / temperature

            # Negative similarity
            # If negative_embs is [B, Dim], treat as 1 negative per sample
            if negative_embs.dim() == 2:
                neg_sim = (
                    self.compute_similarity(anchor_emb, negative_embs) / temperature
                )
                neg_sim = neg_sim.unsqueeze(1)  # [B, 1]
            # If negative_embs is [B, N_neg, Dim], compute sim for each
            elif negative_embs.dim() == 3:
                # anchor: [B, 1, Dim]
                anchor_expanded = anchor_emb.unsqueeze(1)
                # neg_sim: [B, N_neg]
                neg_sim = (
                    F.cosine_similarity(anchor_expanded, negative_embs, dim=2)
                    / temperature
                )

            # InfoNCE Loss
            # numerator: e^pos
            numerator = torch.exp(pos_sim)

            # denominator: e^pos + Σe^neg
            denominator = numerator + torch.sum(torch.exp(neg_sim), dim=-1)

            contrastive_loss = -torch.log(numerator / (denominator + 1e-8))
            contrastive_loss = contrastive_loss.mean()

            # Regularization (Squared L2 norm)
            reg_loss = 0.0
            for param in self.projection.parameters():
                reg_loss += torch.sum(param**2)
            reg_loss *= self.lambda_reg

            return contrastive_loss + reg_loss

    class CryptoEmbeddingTrainer:
        """Trainer for contrastive embedding fine-tuning."""

        def __init__(
            self,
            model: ContrastiveEmbeddingModel,
            learning_rate: float = 1e-4,
            batch_size: int = 32,
            device: str = "cpu",
        ):
            self.model = model.to(device)
            self.device = device
            self.batch_size = batch_size

            self.optimizer = torch.optim.Adam(
                model.projection.parameters(), lr=learning_rate
            )

            logger.info(
                f"Trainer initialized: lr={learning_rate}, batch_size={batch_size}"
            )

        def train_epoch(self, dataloader: DataLoader) -> float:
            """Train for one epoch"""
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            # Automatic Mixed Precision to save GPU memory
            scaler = (
                torch.amp.GradScaler(device=self.device)
                if self.device == "cuda"
                else None
            )

            for batch in dataloader:
                self.optimizer.zero_grad()

                with (
                    torch.amp.autocast(device_type=self.device)
                    if scaler
                    else torch.autocast(device_type=self.device, enabled=False)
                ):
                    anchor_emb = self.model.encode(batch["anchor"])
                    positive_emb = self.model.encode(batch["positive"])

                    negatives = batch["negatives"]
                    # Transpose the list of negatives to group by negative index
                    negatives_by_k = list(zip(*negatives))

                    neg_embs_list = []
                    for neg_texts_k in negatives_by_k:
                        # Encode the k-th negative for all samples in the batch
                        neg_embs_list.append(self.model.encode(list(neg_texts_k)))

                    # Stack to [B, N_neg, Dim]
                    negative_embs = torch.stack(neg_embs_list, dim=1)

                    loss = self.model.compute_contrastive_loss(
                        anchor_emb, positive_emb, negative_embs
                    )

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # Proactive garbage collection for extreme OOM cases
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                total_loss += loss.item()
                num_batches += 1

            return total_loss / num_batches

        def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
            """Evaluate embedding quality"""
            self.model.eval()

            pos_similarities = []
            neg_similarities = []

            with torch.no_grad():
                for batch in dataloader:
                    anchor_emb = self.model.encode(batch["anchor"])
                    positive_emb = self.model.encode(batch["positive"])

                    # Handle multiple negatives: Transpose to get [N_neg][B]
                    negatives = batch["negatives"]
                    negatives_by_k = list(zip(*negatives))

                    # Just take the first negative batch for simple evaluation metric
                    first_neg_batch = negatives_by_k[0]
                    negative_emb = self.model.encode(list(first_neg_batch))

                    pos_sim = self.model.compute_similarity(anchor_emb, positive_emb)
                    neg_sim = self.model.compute_similarity(anchor_emb, negative_emb)

                    pos_similarities.extend(pos_sim.cpu().numpy().tolist())
                    neg_similarities.extend(neg_sim.cpu().numpy().tolist())

            pos_similarities = np.array(pos_similarities)
            neg_similarities = np.array(neg_similarities)

            metrics = {
                "mean_pos_similarity": np.mean(pos_similarities),
                "mean_neg_similarity": np.mean(neg_similarities),
                "pos_neg_gap": np.mean(pos_similarities) - np.mean(neg_similarities),
                "separation_rate": np.mean(pos_similarities > neg_similarities),
                "target_pos_distance": 1 - np.mean(pos_similarities),
                "target_neg_distance": 1 - np.mean(neg_similarities),
            }

            return metrics

else:
    # Dummy classes when torch is not available
    class CryptoEmbeddingDataset:
        """Placeholder when PyTorch is not available"""

        def __init__(self, *args, **kwargs):
            logger.warning("CryptoEmbeddingDataset requires PyTorch")

    class ContrastiveEmbeddingModel:
        """Placeholder when PyTorch is not available"""

        def __init__(self, *args, **kwargs):
            logger.warning("ContrastiveEmbeddingModel requires PyTorch")

    class CryptoEmbeddingTrainer:
        """Placeholder when PyTorch is not available"""

        def __init__(self, *args, **kwargs):
            logger.warning("CryptoEmbeddingTrainer requires PyTorch")
