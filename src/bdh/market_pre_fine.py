import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# --- 1. Helper: Rotary Positional Embeddings (RoPE) ---
class RoPE(nn.Module):
    """
    Rotary Positional Embeddings.
    Rotates query and key vectors to encode relative positions.
    """

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        # Precompute theta
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('theta', theta)

        # Cache for sin/cos
        self.register_buffer('cos_cached', None)
        self.register_buffer('sin_cached', None)

    def _update_cache(self, seq_len: int, device: torch.device):
        if (self.cos_cached is None) or (self.cos_cached.shape[0] < seq_len):
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(t, self.theta)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Time, Dim] or [Batch, Heads, Time, Dim_Head]
        # We assume operations on the last dimension
        batch, heads, seq_len, d = x.shape
        self._update_cache(seq_len, x.device)

        # Slice cache to sequence length
        cos = self.cos_cached[:seq_len].view(1, 1, seq_len, d)
        sin = self.sin_cached[:seq_len].view(1, 1, seq_len, d)

        # Rotate
        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]
        return torch.cat((-x2, x1), dim=-1) * sin + x * cos


# --- 2. Core BDH Components (Appendix E Implementation) ---
class LinearAttention(nn.Module):
    def __init__(self, d_head: int):
        super().__init__()
        self.rope = RoPE(d_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k: [Batch, Heads, Time, Neuron_Dim_Head] (Actually N//H)
        # v:    [Batch, 1, Time, d_model]

        # Apply RoPE to Queries and Keys (in Neuron Dimension)
        # Note: The paper applies RoPE in the N-dimension (neuron index)
        # but typically RoPE is Time. The code in App E implies RoPE(Q)
        # where Q=x (activations).
        # Here we interpret the sequence dimension for RoPE as the Time dimension T.

        qr = self.rope(q)
        kr = self.rope(k)

        # Linear Attention: (Q @ K.T) @ V
        # With causal masking (tril)

        # Dimensions:
        # Q, K: [B, H, T, N_sub]
        # V:    [B, 1, T, D]

        # To make this efficient L(T) not T^2, we use the recurrent form or simple masking for short ctx.
        # Appendix E implementation uses direct T^2 multiplication with masking:
        # return (Qr @ Kr.mT).tril(diagonal=-1) @ V

        # We need to align dimensions for matmul.
        # Q @ K.T -> [B, H, T, T] (Attention Scores)
        attn_scores = torch.matmul(qr, kr.transpose(-2, -1))

        # Causal Mask
        mask = torch.tril(torch.ones_like(attn_scores), diagonal=-1)
        # Note: App E uses diagonal=-1 (strictly causal, next token prediction)
        attn_scores = attn_scores * mask

        # Apply to V
        # V needs to broadcast over H
        # [B, H, T, T] @ [B, 1, T, D] -> [B, H, T, D]
        out = torch.matmul(attn_scores, v)

        return out


class BDH_Layer(nn.Module):
    def __init__(self, d_model: int, n_neurons: int, n_heads: int, dropout: float = 0.05):
        super().__init__()
        self.d = d_model
        self.n = n_neurons
        self.h = n_heads

        self.ln = nn.LayerNorm(d_model, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(dropout)

        # Decoder: d -> n (Projects Latent State to Neurons)
        self.decoder_x = nn.Parameter(
            torch.zeros((n_heads, d_model, n_neurons // n_heads)).normal_(std=0.02)
        )
        self.decoder_y = nn.Parameter(
            torch.zeros((n_heads, d_model, n_neurons // n_heads)).normal_(std=0.02)
        )

        # Encoder: n -> d (Projects Neurons back to Latent State)
        self.encoder = nn.Parameter(
            torch.zeros((n_neurons, d_model)).normal_(std=0.02)
        )

        self.attn = LinearAttention(d_model)  # RoPE dimension managed internally

    def forward(self, v_ast: torch.Tensor) -> torch.Tensor:
        # v_ast (Input Value/State): [Batch, 1, Time, D]
        B, _, T, D = v_ast.shape
        H, N_sub = self.h, self.n // self.h

        # 1. Expand Latent State to Neurons (Excitation)
        # v_ast @ decoder_x -> [B, 1, T, D] @ [H, D, N_sub] -> [B, H, T, N_sub]
        # We need to broadcast v_ast over H
        v_in = v_ast.transpose(1, 2)  # [B, T, 1, D]

        # Manually broadcasting matmul
        # decoder_x: [H, D, N_sub] -> [1, H, D, N_sub]
        dec_x = self.decoder_x.unsqueeze(0)

        # [B, 1, T, D] @ [1, H, D, N_sub] is tricky.
        # Let's align: [B, T, 1, D] @ [1, 1, D, H*N_sub] ? No, heads are separate.

        # Einsum is cleanest here:
        # b: batch, t: time, d: dim, h: head, n: neuron_sub
        x = torch.einsum('b1td,hdn->bhtn', v_ast, self.decoder_x)
        x = F.relu(x)  # Positive Orthant Activation

        # 2. Linear Attention (Neurons Interacting)
        # Q=x, K=x, V=v_ast
        # Output: [B, H, T, D]
        a_ast = self.attn(q=x, k=x, v=v_ast)

        # 3. Compute Y (Modulation)
        # v_ln = LayerNorm(a_ast) inside Einsum logic?
        # App E: y = F.relu(self.ln(a_ast) @ self.decoder_y) * x

        # Apply LN to attention output
        a_ast_norm = self.ln(a_ast)  # [B, H, T, D]

        # Expand Y
        y = torch.einsum('bhtd,hdn->bhtn', a_ast_norm, self.decoder_y)
        y = F.relu(y)

        # Gating / Hebbian-like update
        y = y * x
        y = self.drop(y)

        # 4. Compress back to Latent State (Inhibition/Aggregation)
        # y: [B, H, T, N_sub]
        # encoder: [N, D] -> reshaped to [H, N_sub, D]
        enc = self.encoder.view(H, N_sub, D)

        # [B, H, T, N_sub] @ [H, N_sub, D] -> [B, H, T, D] -> sum over H -> [B, T, D]
        delta_v = torch.einsum('bhtn,hnd->btd', y, enc)

        # Residual Connection
        v_out = v_ast.squeeze(1) + delta_v  # [B, T, D]

        # Final LayerNorm
        v_out = self.ln(v_out)

        return v_out.unsqueeze(1)  # Restore [B, 1, T, D]


# --- 3. Market BDH Model (The Wrapper) ---
class MarketBDH(nn.Module):
    def __init__(
            self,
            num_assets: int,
            num_features: int,
            d_model: int = 256,
            n_neurons: int = 8192,
            n_heads: int = 4,
            n_layers: int = 4,
            mode: str = 'pretrain'
    ):
        """
        BDH Model adapted for Market Dynamics.

        Args:
            num_assets: Number of stocks/assets (N)
            num_features: Number of features per asset (F)
            d_model: Latent driver dimension (D)
            n_neurons: Number of neurons in the brain (Should be >= num_assets)
            mode: 'pretrain' (predict features) or 'finetune' (predict return)
        """
        super().__init__()
        self.num_assets = num_assets
        self.num_features = num_features
        self.mode = mode

        # 1. Input Adapter (The "Eyes")
        # Maps the full market state [N, F] into the latent driver [D]
        # We use a linear projection that effectively learns "Global Market Factors"
        self.input_adapter = nn.Linear(num_assets * num_features, d_model)

        # Asset Identity Embeddings (Optional, if we want to add specific noise/bias)
        # For this architecture, the input_adapter handles identity mapping.

        # 2. The BDH Brain (Physics Engine)
        self.layers = nn.ModuleList([
            BDH_Layer(d_model, n_neurons, n_heads) for _ in range(n_layers)
        ])

        # 3. Heads (The "Mouth")
        # Pretraining Head: Predicts next step features for ALL assets [D -> N*F]
        self.head_pretrain = nn.Linear(d_model, num_assets * num_features)

        # Finetuning Head: Predicts next step binary direction for ALL assets [D -> N]
        self.head_finetune = nn.Linear(d_model, num_assets)

    def set_mode(self, mode: str):
        if mode not in ['pretrain', 'finetune']:
            raise ValueError("Mode must be 'pretrain' or 'finetune'")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Time, Num_Assets, Num_Features]

        Returns:
            If pretrain: [Batch, Time, Num_Assets, Num_Features] (Next step pred)
            If finetune: [Batch, Time, Num_Assets] (Probabilities 0-1)
        """
        B, T, N, F = x.shape

        # 1. Flatten Inputs: [B, T, N*F]
        x_flat = x.view(B, T, N * F)

        # 2. Encode to Latent Driver: [B, T, D]
        # This v_ast represents the "Global Market State" driving the neurons
        v_ast = self.input_adapter(x_flat)
        v_ast = v_ast.unsqueeze(1)  # [B, 1, T, D] required for BDH Layer

        # 3. Pass through Layers (Recurrent Reasoning)
        for layer in self.layers:
            v_ast = layer(v_ast)

        # v_ast is now [B, 1, T, D] - The predicted latent state for the next step
        state = v_ast.squeeze(1)  # [B, T, D]

        # 4. Output Head
        if self.mode == 'pretrain':
            # Regression: Predict features
            out = self.head_pretrain(state)  # [B, T, N*F]
            return out.view(B, T, N, F)

        else:  # finetune
            # Classification: Predict probabilities
            logits = self.head_finetune(state)  # [B, T, N]
            return torch.sigmoid(logits)


# --- Example Usage ---
if __name__ == "__main__":
    # Hyperparameters
    B, T, N, F = 2, 64, 100, 20
    D, Neurons = 256, 1024

    # Random Data (Positive Orthant Mock)
    x = torch.abs(torch.randn(B, T, N, F))

    # Initialize Model
    model = MarketBDH(num_assets=N, num_features=F, d_model=D, n_neurons=Neurons, mode='pretrain')

    # 1. Pretraining Pass
    pred_feats = model(x)
    print(f"Pretrain Output: {pred_feats.shape} (Should be [{B}, {T}, {N}, {F}])")

    # 2. Switch to Finetuning
    model.set_mode('finetune')
    pred_probs = model(x)
    print(f"Finetune Output: {pred_probs.shape} (Should be [{B}, {T}, {N}])")