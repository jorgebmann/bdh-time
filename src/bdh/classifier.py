# Classification wrapper for the BDH model
# Adapts the language model for text classification tasks

import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import BDH, BDHConfig


class BDHClassifier(nn.Module):
    """
    Wrapper class that adapts the BDH model for classification tasks.

    The BDH model is used as a feature extractor, and a linear classification
    head is added on top. The sequence representations are pooled (mean pooling)
    before being fed to the classifier.
    """

    def __init__(self, config: BDHConfig, num_classes: int = 2):
        """
        Initialize the BDH classifier.

        Args:
            config: BDHConfig object with model hyperparameters
            num_classes: Number of output classes (default: 2 for binary classification)
        """
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # The BDH model as feature extractor
        self.bdh_core = self._build_bdh_core(config)

        # Removed: Learnable Pooling Query (caused overfitting on small data)
        # self.pooling_query = nn.Parameter(torch.zeros(config.n_embd).normal_(std=0.02))

        # Classification head
        self.classifier = nn.Linear(config.n_embd, num_classes)

        # Initialize the classifier head
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        # Label smoothing parameter
        self.label_smoothing = 0.1

    def _build_bdh_core(self, config: BDHConfig):
        """
        Build the core BDH model.
        """
        bdh_model = BDH(config)
        return bdh_model

    def forward(self, idx, targets=None):
        """
        Forward pass for classification.

        Args:
            idx: Input token indices, shape (B, T)
            targets: Optional target labels, shape (B,)

        Returns:
            logits: Classification logits, shape (B, num_classes)
            loss: Cross-entropy loss if targets provided, else None
        """
        # Get the BDH representations

        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # Run through the BDH core (embedding + layers)
        x = self.bdh_core.embed(idx).unsqueeze(1)  # (B, 1, T, D)
        x = self.bdh_core.ln(x)

        # Process through layers
        for _ in range(C.n_layer):
            x_res = x

            # Project to conceptual space
            x_latent = x @ self.bdh_core.encoder
            x_sparse = F.relu(x_latent)  # (B, nh, T, N)

            # Attention in conceptual space
            # UPDATE: Use Bidirectional Attention (is_causal=False)
            yKV = self.bdh_core.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
                is_causal=False  # <--- CRITICAL UPDATE FOR CLASSIFICATION
            )
            yKV = self.bdh_core.ln(yKV)

            # Modulation
            y_latent = yKV @ self.bdh_core.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.bdh_core.drop(xy_sparse)

            # Project back to working space
            yMLP = (
                    xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.bdh_core.decoder
            )
            y = self.bdh_core.ln(yMLP)

            # Apply residual connection
            x = self.bdh_core.ln(x_res + y)

        # Now x has shape (B, 1, T, D)
        # Remove the head dimension and get (B, T, D)
        x = x.squeeze(1)  # (B, T, D)

        # UPDATE: Mean Pooling
        # Assuming 50256 is the padding token (standard for GPT-2/tiktoken)
        # We create a mask to ignore padding in the average
        padding_token = 50256
        mask = (idx != padding_token).float().unsqueeze(-1)  # (B, T, 1)

        # Sum valid embeddings
        sum_embeddings = (x * mask).sum(dim=1)  # (B, D)

        # Count valid tokens (prevent div by zero)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)

        # Calculate Mean
        pooled = sum_embeddings / sum_mask  # (B, D)

        # Classification head
        logits = self.classifier(pooled)  # (B, num_classes)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

        return logits, loss

    def predict(self, idx):
        """
        Convenience method for getting predictions.

        Args:
            idx: Input token indices, shape (B, T)

        Returns:
            predictions: Predicted class indices, shape (B,)
            probabilities: Class probabilities, shape (B, num_classes)
        """
        logits, _ = self.forward(idx)
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        return predictions, probabilities
