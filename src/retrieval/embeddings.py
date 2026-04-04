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


# PyTorch-based classes (only available if torch is installed)
if TORCH_AVAILABLE:

    class ContrastiveEmbeddingModel(nn.Module):
        """
        Embedding model with contrastive fine-tuning capability.
        Implements Equation (7):
        L = -log(e^s+ / (e^s+ + Σe^s-)) + λ||θ||²
        """

        def __init__(
            self,
            base_model_name: str = "bge-vi-base",
            embedding_dim: int = None,
            lambda_reg: float = 0.001,
            freeze_base: bool = True,
            max_length: int = 256,
            encoder_device: str = "cpu",
            encode_batch_size: int = 16,
        ):
            super().__init__()

            self.lambda_reg = lambda_reg
            import os

            self.embedding_dim = (
                embedding_dim
                if embedding_dim is not None
                else int(os.getenv("RETRIEVER_EMBEDDING_DIM", "1024"))
            )
            # Strict memory cap on attention matrices
            self.max_length = max_length
            self.encoder_device = encoder_device
            self.encode_batch_size = encode_batch_size

            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.encoder = SentenceTransformer(
                    base_model_name, device=encoder_device
                )

                if hasattr(self.encoder, "max_seq_length"):
                    self.encoder.max_seq_length = max_length

                # CRITICAL: auto-cap max_length at the model's hard limit.
                # BERT-based models (bge-small etc.) have exactly 512 position
                # embeddings — passing anything larger causes a guaranteed crash.
                try:
                    model_max_pos = self.encoder[
                        0
                    ].auto_model.config.max_position_embeddings
                    if max_length > model_max_pos:
                        logger.warning(
                            f"--max_length {max_length} exceeds model's "
                            f"max_position_embeddings ({model_max_pos}). "
                            f"Auto-capping to {model_max_pos}."
                        )
                        max_length = model_max_pos
                        self.max_length = max_length
                except Exception:
                    pass  # Best-effort; proceed with user-supplied value

                # Propagate capped value to all tokenizer layers
                try:
                    if hasattr(self.encoder[0], "max_seq_length"):
                        self.encoder[0].max_seq_length = max_length
                    if hasattr(self.encoder[0], "tokenizer"):
                        self.encoder[0].tokenizer.model_max_length = max_length
                except Exception:
                    pass

                # Explicitly enforce device placement before anything else
                self.encoder.to(encoder_device)

                # If doing long sequences, enable gradient checkpointing to save VRAM
                if max_length > 512 and not freeze_base:
                    if hasattr(
                        self.encoder[0].auto_model, "gradient_checkpointing_enable"
                    ):
                        logger.warning(
                            f"Max length {max_length} is large! Enabling Gradient Checkpointing to save VRAM."
                        )
                        self.encoder[0].auto_model.gradient_checkpointing_enable()

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

        def to(self, *args, **kwargs):
            """Override to() to prevent encoder from moving to GPU when freeze_base=True"""
            # First, move everything to the requested device normally
            super().to(*args, **kwargs)

            # Then gracefully pin the encoder back to the chosen encoder_device
            # if we are doing cpu-microbatching.
            if getattr(self, "encoder", None) is not None:
                self.encoder.to(self.encoder_device)

            return self

        def _encode_base_microbatch(self, texts: List[str]) -> torch.Tensor:
            """
            Encode base embeddings in micro-batches to control peak memory.
            Runs on self.encoder_device (CPU recommended).
            """
            if self.encoder is None:
                return torch.randn(len(texts), self.embedding_dim)

            # If we are training and base is not frozen, we need gradients
            if self.training and any(
                p.requires_grad for p in self.encoder.parameters()
            ):
                device = next(self.projection.parameters()).device
                # Tokenize with strict mapping to prevent OOM
                features = self.encoder.tokenize(texts)

                if hasattr(self.encoder.tokenizer, "pad_token_id"):
                    features = self.encoder.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )

                features = {k: v.to(device) for k, v in features.items()}
                # Forward pass through base model to get gradients
                out_features = self.encoder(features)
                return out_features["sentence_embedding"]

            # Ordinary frozen execution (micro-batched on CPU for safety)
            outs = []
            # Get the actual HuggingFace tokenizer for guaranteed truncation.
            # ST's encoder.tokenize() sometimes ignores max_seq_length, so we
            # call the underlying tokenizer directly with explicit truncation args.
            try:
                hf_tokenizer = self.encoder[0].tokenizer
            except (IndexError, AttributeError):
                hf_tokenizer = getattr(self.encoder, "tokenizer", None)

            with torch.no_grad():
                for i in range(0, len(texts), self.encode_batch_size):
                    chunk = texts[i : i + self.encode_batch_size]

                    if hf_tokenizer is not None:
                        # Use HF tokenizer directly with strict truncation — guaranteed safe
                        features = hf_tokenizer(
                            chunk,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt",
                        )
                    else:
                        # Last resort: rely on ST tokenize() with max_seq_length set above
                        features = self.encoder.tokenize(chunk)

                    features = {
                        k: v.to(self.encoder_device) for k, v in features.items()
                    }
                    out_features = self.encoder.forward(features)

                    if "sentence_embedding" in out_features:
                        emb = out_features["sentence_embedding"]
                    else:
                        # Mean pooling fallback for models without pooling layer
                        token_embs = out_features["token_embeddings"]
                        mask = out_features["attention_mask"].unsqueeze(-1).float()
                        emb = (token_embs * mask).sum(1) / torch.clamp(
                            mask.sum(1), min=1e-9
                        )

                    outs.append(emb)
            return torch.cat(outs, dim=0)

        def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
            """Encode texts to embeddings."""
            proj_device = next(self.projection.parameters()).device

            base_embeddings = self._encode_base_microbatch(texts)
            base_embeddings = base_embeddings.to(proj_device, non_blocking=True)

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

    class RetrievalDataset(TorchDataset):
        """
        Dataset for training a retrieval model on claim-evidence pairs.

        Format of ``records``:
            [
              {"claim": "...", "evidences": ["sentence 1", "sentence 2", ...]},
              ...
            ]

        Each triplet:
          anchor   = claim text
          positive = one randomly sampled evidence sentence from that claim
          negatives = evidence sentences sampled from *other* claims

        This is a drop-in replacement for CryptoEmbeddingDataset when the
        training data comes from finfact_raw_truefalse.csv (claim / evidence
        columns) rather than the scam/legit split used by CryptoEmbeddingDataset.
        """

        def __init__(
            self,
            records: List[Dict],
            num_triplets: int = 8000,
            num_negatives: int = 3,
        ):
            """
            Args:
                records: list of dicts with keys "claim" and "evidences"
                num_triplets: total samples __len__ reports (lazy generation)
                num_negatives: how many negative evidence sentences per anchor
            """
            self.records = records
            self.num_triplets = num_triplets
            self.num_negatives = num_negatives

            # Pre-build a flat list of (record_idx, sentence) for fast negative sampling
            self._all_sentences: List[tuple] = []  # (record_idx, sentence)
            for i, rec in enumerate(records):
                for ev in rec["evidences"]:
                    self._all_sentences.append((i, ev))

            logger.info(
                f"RetrievalDataset | records={len(records)} | "
                f"total_sentences={len(self._all_sentences)} | "
                f"num_triplets={num_triplets}"
            )

        def __len__(self) -> int:
            return self.num_triplets

        def __getitem__(self, idx: int) -> Dict[str, Union[str, List[str]]]:
            # Pick a random record as anchor
            anchor_idx = np.random.randint(0, len(self.records))
            anchor_rec = self.records[anchor_idx]

            # Anchor = claim text
            anchor_text = anchor_rec["claim"]

            # Positive = one of its evidence sentences
            positive_text = anchor_rec["evidences"][
                np.random.randint(0, len(anchor_rec["evidences"]))
            ]

            # Negatives = evidence sentences from *other* records
            negatives: List[str] = []
            attempts = 0
            while (
                len(negatives) < self.num_negatives
                and attempts < self.num_negatives * 20
            ):
                attempts += 1
                neg_global_idx = np.random.randint(0, len(self._all_sentences))
                neg_rec_idx, neg_sentence = self._all_sentences[neg_global_idx]
                if neg_rec_idx != anchor_idx:
                    negatives.append(neg_sentence)

            # Safety: if we couldn't find enough, reuse what we have
            while len(negatives) < self.num_negatives:
                negatives.append(negatives[0] if negatives else positive_text)

            return {
                "anchor": anchor_text,
                "positive": positive_text,
                "negatives": negatives,
            }

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
            import time as _time

            self.model.train()
            total_loss = 0.0
            num_batches = 0
            total_steps = len(dataloader)
            log_every = max(1, min(50, total_steps // 5))  # log ~5 times per epoch

            # Gradient Accumulation Steps
            accum_steps = 4

            # Automatic Mixed Precision to save GPU memory
            scaler = (
                torch.amp.GradScaler(device=self.device)
                if self.device == "cuda"
                else None
            )

            logger.info(
                f"[train_epoch] {total_steps} steps | accum={accum_steps} | "
                f"effective_batch={self.batch_size * accum_steps} | device={self.device}"
            )
            t_epoch_start = _time.time()

            for step, batch in enumerate(dataloader):
                # 1. Prepare flat text list to encode everything in ONE pass
                anchors: List[str] = batch["anchor"]
                positives: List[str] = batch["positive"]
                negatives: List[List[str]] = batch["negatives"]  # [B][N_neg]

                B = len(anchors)
                N_neg = len(negatives[0]) if B > 0 else 0

                # Flatten negatives: [B * N_neg]
                neg_flat = [negatives[i][j] for i in range(B) for j in range(N_neg)]

                # Combine all texts
                all_texts = anchors + positives + neg_flat

                with (
                    torch.amp.autocast(device_type=self.device)
                    if scaler
                    else torch.autocast(device_type=self.device, enabled=False)
                ):
                    # One big encode: [2B + B*N_neg, Dim]
                    all_embs = self.model.encode(all_texts)

                    # Extract slices
                    anchor_emb = all_embs[0:B]
                    positive_emb = all_embs[B : 2 * B]
                    neg_emb_flat = all_embs[2 * B :]
                    negative_embs = neg_emb_flat.view(B, N_neg, -1)  # [B, N_neg, Dim]

                    # Compute loss
                    loss = self.model.compute_contrastive_loss(
                        anchor_emb, positive_emb, negative_embs
                    )

                    # Scale down loss for gradient accumulation
                    loss = loss / accum_steps

                if scaler:
                    scaler.scale(loss).backward()

                    if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        logger.debug(
                            f"  [step {step + 1}] grad update | loss={float(loss.item()) * accum_steps:.4f}"
                        )
                else:
                    loss.backward()
                    if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        logger.debug(
                            f"  [step {step + 1}] grad update | loss={float(loss.item()) * accum_steps:.4f}"
                        )

                batch_loss = float(loss.item()) * accum_steps
                total_loss += batch_loss
                num_batches += 1

                # Log progress every log_every steps
                if (step + 1) % log_every == 0 or (step + 1) == total_steps:
                    elapsed = _time.time() - t_epoch_start
                    avg_so_far = total_loss / num_batches
                    steps_per_sec = (step + 1) / max(elapsed, 1e-6)
                    eta = (total_steps - step - 1) / max(steps_per_sec, 1e-6)
                    logger.info(
                        f"  step {step + 1:>4}/{total_steps} | "
                        f"loss={batch_loss:.4f} | avg={avg_so_far:.4f} | "
                        f"elapsed={elapsed:.1f}s | ETA={eta:.1f}s"
                    )

            epoch_time = _time.time() - t_epoch_start
            avg_loss = total_loss / max(num_batches, 1)
            logger.info(
                f"[train_epoch done] avg_loss={avg_loss:.4f} | "
                f"batches={num_batches} | time={epoch_time:.1f}s"
            )
            return avg_loss

        def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
            """Evaluate embedding quality"""
            self.model.eval()
            logger.info("[evaluate] Running evaluation on dataloader...")

            pos_similarities = []
            neg_similarities = []
            total_eval_steps = len(dataloader)

            with torch.no_grad():
                for eval_step, batch in enumerate(dataloader):
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

                    if (eval_step + 1) % max(1, total_eval_steps // 4) == 0:
                        logger.debug(
                            f"  [eval] step {eval_step + 1}/{total_eval_steps} | "
                            f"samples so far: {len(pos_similarities)}"
                        )

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

            logger.info(
                f"\n[evaluate results]\n"
                f"  Samples evaluated  : {len(pos_similarities)}\n"
                f"  mean_pos_similarity: {metrics['mean_pos_similarity']:.4f}  (muốn -> 1.0)\n"
                f"  mean_neg_similarity: {metrics['mean_neg_similarity']:.4f}  (muốn -> 0.0)\n"
                f"  pos_neg_gap        : {metrics['pos_neg_gap']:.4f}  (muốn tăng)\n"
                f"  separation_rate    : {metrics['separation_rate']:.4f}  (% pos > neg, muốn -> 1.0)\n"
                f"  target_pos_dist    : {metrics['target_pos_distance']:.4f}\n"
                f"  target_neg_dist    : {metrics['target_neg_distance']:.4f}"
            )
            return metrics

else:
    # Dummy classes when torch is not available
    class RetrievalDataset:
        """Placeholder when PyTorch is not available"""

        def __init__(self, *args, **kwargs):
            logger.warning("RetrievalDataset requires PyTorch")

    class ContrastiveEmbeddingModel:
        """Placeholder when PyTorch is not available"""

        def __init__(self, *args, **kwargs):
            logger.warning("ContrastiveEmbeddingModel requires PyTorch")

    class CryptoEmbeddingTrainer:
        """Placeholder when PyTorch is not available"""

        def __init__(self, *args, **kwargs):
            logger.warning("CryptoEmbeddingTrainer requires PyTorch")
