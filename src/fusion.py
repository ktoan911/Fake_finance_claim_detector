"""
Confidence-Aware Fusion Layer

Implements Section 3.2 of the paper:
- Combines LM logits pLM(y|q) with retrieval evidence pret(y|D)
- pfinal(y|q, D) = σ(β · pLM + (1 − β) · MLP(pret))
- β is trainable gating parameter (initialized at 0.5)
- Contrastive loss training with regularization
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
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
        confidence: torch.Tensor

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
            output_dim: int = 2,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
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
            learn_beta: bool = True
        ):
            super().__init__()
            
            self.num_classes = num_classes
            self.lambda_reg = lambda_reg
            
            # Trainable gating parameter β
            self._beta_logit = nn.Parameter(
                torch.tensor(self._inverse_sigmoid(initial_beta)),
                requires_grad=learn_beta
            )
            
            # MLP for projecting retrieval scores
            self.retrieval_mlp = RetrievalMLP(
                input_dim=retrieval_input_dim,
                hidden_dim=hidden_dim,
                output_dim=num_classes
            )
            
            # Confidence estimation head
            self.confidence_head = nn.Sequential(
                nn.Linear(num_classes * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
            logger.info(f"ConfidenceAwareFusion initialized: β={initial_beta}, λ={lambda_reg}")
        
        def _inverse_sigmoid(self, x: float) -> float:
            """Inverse sigmoid for initialization"""
            x = np.clip(x, 1e-6, 1 - 1e-6)
            return np.log(x / (1 - x))
        
        @property
        def beta(self) -> torch.Tensor:
            """Get the current gating parameter β"""
            return torch.sigmoid(self._beta_logit)
        
        def forward(
            self,
            lm_logits: torch.Tensor,
            retrieval_features: torch.Tensor
        ) -> FusionOutput:
            """Forward pass implementing Equation (2)."""
            # Get current beta
            beta = self.beta
            
            # Project retrieval features to label space
            retrieval_logits = self.retrieval_mlp(retrieval_features)
            
            # Fuse according to Equation (2)
            fused_logits = beta * lm_logits + (1 - beta) * retrieval_logits
            final_probs = torch.sigmoid(fused_logits)
            
            # Estimate confidence
            combined_features = torch.cat([lm_logits, retrieval_logits], dim=-1)
            confidence = self.confidence_head(combined_features)
            
            return FusionOutput(
                final_probs=final_probs,
                fused_logits=fused_logits,
                lm_weight=beta.item(),
                retrieval_weight=(1 - beta).item(),
                confidence=confidence
            )
        
        def compute_contrastive_loss(
            self,
            positive_scores: torch.Tensor,
            negative_scores: torch.Tensor,
            temperature: float = 1.0
        ) -> torch.Tensor:
            """Compute contrastive loss from Equation (3)."""
            sp = positive_scores / temperature
            sn = negative_scores / temperature
            
            numerator = torch.exp(sp)
            denominator = numerator + torch.sum(torch.exp(sn), dim=-1, keepdim=True)
            
            contrastive_loss = -torch.log(numerator / (denominator + 1e-8))
            contrastive_loss = contrastive_loss.mean()
            
            beta_reg = self.lambda_reg * (self.beta ** 2)
            
            return contrastive_loss + beta_reg

    class RetrievalFeatureEncoder(nn.Module):
        """Encodes retrieval results into features for fusion."""
        
        def __init__(
            self,
            num_retrieved: int = 5,
            score_features: int = 4,
            hidden_dim: int = 64,
            output_dim: int = 64
        ):
            super().__init__()
            
            self.num_retrieved = num_retrieved
            input_dim = num_retrieved * score_features
            
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
            self.attention = nn.Sequential(
                nn.Linear(score_features, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            )
        
        def forward(
            self,
            retrieval_scores: torch.Tensor,
            retrieval_features: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            batch_size = retrieval_scores.size(0)
            
            attn_logits = self.attention(retrieval_scores)
            attn_weights = F.softmax(attn_logits, dim=1)
            
            weighted = retrieval_scores * attn_weights
            flat = weighted.view(batch_size, -1)
            encoded = self.encoder(flat)
            
            return encoded

