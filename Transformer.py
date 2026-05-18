import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    """

    # <--- MODIFICATION 1: 接收 d_model 和 d_ff (内部维度)
    def __init__(self, d_model):
        """
        Args:
            d_model (int): Dimension of the input features.
            d_ff (int): Dimension of the internal feed-forward layer.
        """
        super().__init__()
        # 使用 d_ff 作为中间维度
        self.WG = nn.Linear(d_model, d_model * 2)
        self.W1 = nn.Linear(d_model, d_model * 2)
        self.W2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        """
        Forward pass for SwiGLU.
        """
        # (B, N, d_model) -> (B, N, d_ff)
        g = F.silu(self.WG(x))
        # (B, N, d_model) -> (B, N, d_ff)
        z = self.W1(x)
        # (B, N, d_ff) * (B, N, d_ff) -> (B, N, d_ff)
        # (B, N, d_ff) -> (B, N, d_model)
        return self.W2(g * z)


class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, d_model, num_heads, lambda_init):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)

        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.d_head))

        self.lambda_init = lambda_init

        self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self.eps = 1e-5

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.rms_scale, 1.0)

    # <--- MODIFICATION 2: get padding mask，remove causal mask
    def forward(self, X, mask=None):  # 1. 接受 mask
        """
        Forward pass for Multi-Head Differential Attention.

        Args:
            X (Tensor): Input tensor of shape (batch, sequence_length, d_model).
            mask (Tensor, optional): Padding mask of shape (batch, sequence_length).

        Returns:
            Tensor: Output tensor after applying differential attention.
        """
        batch, N, d_model = X.shape

        # ... (Q, K, V, Q1, Q2, K1, K2, lambda_val 的计算不变) ...
        Q = self.W_q(X).view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = self.W_k(X).view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = self.W_v(X).view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)

        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)

        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # ------------------- remove casual mask ------------------- #
        # mask = torch.tril(torch.ones((N, N), device=X.device))...
        # -------------------------------------------------------------------- #

        # +++++++++++++++ 2. mask (Padding Mask) 逻辑 +++++++++++++++ #
        final_mask = None
        if mask is not None:
            # mask shape: (batch, N)
            padding_mask = mask.unsqueeze(1).unsqueeze(2)  # 形状变为 (batch, 1, 1, N)
            # mask True (1) other -inf
            final_mask = padding_mask.masked_fill(padding_mask == 1, float('-inf')).masked_fill(padding_mask == 0, 0.0)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        # Compute attention scores
        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling

        # 3. mask
        if final_mask is not None:
            A1 = A1 + final_mask
            A2 = A2 + final_mask

        # ... (后续计算不变) ...
        attention1 = F.softmax(A1, dim=-1)
        attention2 = F.softmax(A2, dim=-1)
        attention = attention1 - lambda_val * attention2

        O = torch.matmul(attention, V)

        O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head)
        O_normalized = O_normalized * (1 - self.lambda_init)
        O_concat = O_normalized.transpose(1, 2).contiguous().view(batch, N, self.num_heads * 2 * self.d_head)

        out = self.W_o(O_concat)

        return out


class DiffTransformerLayer(nn.Module):
    """
    Single Layer of the DiffTransformer Architecture.
    """

    # <--- MODIFICATION 3: 接收 d_model 和 d_ff (内部维度)
    def __init__(self, d_model, num_heads, lambda_init, dropout_rate=0.1):
        """
        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the internal FFN.
            lambda_init (float): Initial value for lambda.
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadDifferentialAttention(d_model, num_heads, lambda_init)
        self.norm2 = RMSNorm(d_model)
        # 将 d_model 和 d_ff 传递给 SwiGLU
        self.ff = SwiGLU(d_model)
        self.attn_dropout = nn.Dropout(dropout_rate)  # 你原来的代码有 dropout，但没用上，我加回来了
        self.ff_dropout = nn.Dropout(dropout_rate)  # 你原来的代码有 dropout，但没用上，我加回来了

    def forward(self, x, mask=None):  # 1. 接受 mask
        """
        Forward pass for a single transformer layer.
        """
        # 2. 传递 mask，并应用 dropout
        y = self.attn_dropout(self.attn(self.norm1(x), mask=mask)) + x
        # 3. 应用 dropout
        z = self.ff_dropout(self.ff(self.norm2(y))) + y
        return z


class DiffTransformer(nn.Module):
    """
    The DiffTransformer Model.
    """

    # <--- MODIFICATION 4: 接收 d_model, d_ff, 并修改 forward
    def __init__(self, d_model, num_heads, num_layers, dropout_rate=0.1):
        """
        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            dim_feedforward (int): Dimension of the internal FFN.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.layers = nn.ModuleList([
            DiffTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                lambda_init=0.8 - 0.6 * math.exp(-0.3 * (l - 1)),
                dropout_rate=dropout_rate
            )
            for l in range(1, num_layers + 1)
        ])
        self.norm = RMSNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def forward(self, x, mask=None):  # 2. 接受 mask
        """
        Forward pass for the DiffTransformer.

        Args:
            x (Tensor): Input tensor of *embeddings* shape (batch, sequence_length, d_model).
            mask (Tensor, optional): Padding mask of shape (batch, sequence_length).

        Returns:
            Tensor: Output tensor of shape (batch, sequence_length, d_model).
        """
        # batch, N = x.shape
        # positions = torch.arange(N, device=x.device).unsqueeze(0).expand(batch, N)
        # X = self.token_emb(x) + self.pos_emb(positions)
        X = x

        for layer in self.layers:
            X = layer(X, mask=mask)  # 4. mask

        X = self.norm(X)
        return X