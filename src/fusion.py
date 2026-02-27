from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using numpy-based fusion only.")

# PyTorch-based models (only available if torch is installed)
if TORCH_AVAILABLE:

    @dataclass
    class FusionInput:
        """Input container for fusion layer"""

        lm_logits: torch.Tensor
        retrieval_scores: torch.Tensor
        retrieval_features: Optional[torch.Tensor] = None

    @dataclass
    class FusionOutput:
        """Output container for fusion layer"""

        final_probs: torch.Tensor
        fused_logits: torch.Tensor
        lm_weight: float
        retrieval_weight: float

    class RetrievalMLP(nn.Module):
        """
        Two-layer MLP that projects retrieval scores to label space.
        As described in paper: "MLP is a two-layer network that projects
        retrieval scores to the label space"
        """

        def __init__(
            self,
            input_dim: int = 64,
            hidden_dim: int = 128,
            output_dim: int = 1,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

            self._init_weights()

        def _init_weights(self):
            """Initialize weights using Xavier initialization"""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layers(x)

    class ConfidenceAwareFusion(nn.Module):
        """
        Implements Equation (2) from the paper:
        pfinal(y|q, D) = σ(β · pLM + (1 − β) · MLP(pret))

        Where σ is Sigmoid for binary classification (num_classes=2).
        For binary: MLP outputs 1 logit, sigmoid gives P(True), P(False)=1-P(True)

        With contrastive loss from Equation (3):
        L = -log(e^sp / Σe^sn) + λ||β||²
        """

        def __init__(
            self,
            retrieval_input_dim: int = 64,
            hidden_dim: int = 128,
            num_classes: int = 2,
            initial_beta: float = 0.5,
            lambda_reg: float = 0.01,
            learn_beta: bool = True,
        ):
            super().__init__()

            self.num_classes = num_classes
            self.lambda_reg = lambda_reg
            self.is_binary = num_classes == 2

            # Trainable gating parameter β
            # We use a logit parameter and apply sigmoid in forward() to ensure β ∈ [0, 1]
            self._beta_logit = nn.Parameter(
                torch.tensor(self._inverse_sigmoid(initial_beta)),
                requires_grad=learn_beta,
            )

            # MLP for projecting retrieval scores
            # For binary classification, output 1 logit; else output num_classes logits
            mlp_output_dim = 1 if self.is_binary else num_classes
            self.retrieval_mlp = RetrievalMLP(
                input_dim=retrieval_input_dim,
                hidden_dim=hidden_dim,
                output_dim=mlp_output_dim,
            )

            activation_type = "sigmoid" if self.is_binary else "softmax"
            logger.info(
                f"ConfidenceAwareFusion initialized: β={initial_beta}, λ={lambda_reg}, num_classes={num_classes}, activation={activation_type}"
            )

        def _inverse_sigmoid(self, x: float) -> float:
            """Inverse sigmoid for initialization"""
            x = np.clip(x, 1e-6, 1 - 1e-6)
            return np.log(x / (1 - x))

        @property
        def beta(self) -> torch.Tensor:
            """Get the current gating parameter β. Guaranteed to be in [0, 1]."""
            return torch.sigmoid(self._beta_logit)

        def forward(
            self, lm_logits: torch.Tensor, retrieval_features: torch.Tensor
        ) -> FusionOutput:
            """Forward pass implementing Equation (2): β·pLM + (1-β)·MLP(pret)."""

            batch_size = lm_logits.size(0)
            beta = self.beta

            # Project retrieval features to label space
            retrieval_logits = self.retrieval_mlp(retrieval_features)

            if self.is_binary:
                # Binary classification: treat as 2-class softmax (same as multi-class)
                # lm_logits: [B, 2] → [logit_True, logit_False]
                assert lm_logits.size(1) == 2, (
                    f"Binary mode: lm_logits should be [B, 2], got {lm_logits.shape}"
                )

                # For binary, MLP outputs 1 logit → expand to 2 logits [pos, neg]
                # retrieval_logits: [B, 1] → treat as logit_True; logit_False = -logit_True
                assert retrieval_logits.size() == (batch_size, 1), (
                    f"Binary mode: retrieval_logits should be [B, 1], got {retrieval_logits.shape}"
                )

                # Expand retrieval to 2 logits: [logit_True, logit_False] = [r, -r]
                retrieval_logits_2 = torch.cat(
                    [retrieval_logits, -retrieval_logits], dim=-1
                )  # [B, 2]

                # Fuse: β·pLM + (1-β)·MLP(pret)
                fused_logits = (
                    beta * lm_logits + (1 - beta) * retrieval_logits_2
                )  # [B, 2]

                # Apply softmax to get probabilities
                final_probs = torch.softmax(fused_logits, dim=-1)  # [B, 2]
                # final_probs[:, 0] = P(True), final_probs[:, 1] = P(False)

            else:
                # Multi-class: use softmax
                assert lm_logits.size(1) == self.num_classes, (
                    f"lm_logits shape mismatch: expected [B, {self.num_classes}], got {lm_logits.shape}"
                )

                assert retrieval_logits.size() == (batch_size, self.num_classes), (
                    f"retrieval_logits shape mismatch: expected [{batch_size}, {self.num_classes}], got {retrieval_logits.shape}"
                )

                fused_logits = beta * lm_logits + (1 - beta) * retrieval_logits
                final_probs = torch.softmax(fused_logits, dim=-1)

            return FusionOutput(
                final_probs=final_probs,
                fused_logits=fused_logits,
                lm_weight=beta.item(),
                retrieval_weight=(1 - beta).item(),
            )

        def compute_contrastive_loss(
            self,
            positive_scores: torch.Tensor,
            negative_scores: torch.Tensor,
            temperature: float = 1.0,
        ) -> torch.Tensor:
            """Compute contrastive loss from Equation (3)."""
            sp = positive_scores / temperature
            sn = negative_scores / temperature

            numerator = torch.exp(sp)
            denominator = numerator + torch.sum(torch.exp(sn), dim=-1, keepdim=True)

            contrastive_loss = -torch.log(numerator / (denominator + 1e-8))
            contrastive_loss = contrastive_loss.mean()

            beta_reg = self.lambda_reg * (self.beta**2)

            return contrastive_loss + beta_reg

    class RetrievalFeatureEncoder(nn.Module):
        """Encodes retrieval results into features for fusion."""

        def __init__(
            self,
            num_retrieved: int = 5,
            score_features: int = 4,
            hidden_dim: int = 64,
            output_dim: int = 64,
        ):
            super().__init__()

            self.num_retrieved = num_retrieved
            input_dim = num_retrieved * score_features

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

            self.attention = nn.Sequential(
                nn.Linear(score_features, 16), nn.Tanh(), nn.Linear(16, 1)
            )

        def forward(
            self,
            retrieval_scores: torch.Tensor,
            retrieval_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            batch_size = retrieval_scores.size(0)

            attn_logits = self.attention(retrieval_scores)
            attn_weights = F.softmax(attn_logits, dim=1)

            weighted = retrieval_scores * attn_weights
            flat = weighted.view(batch_size, -1)
            encoded = self.encoder(flat)

            return encoded
