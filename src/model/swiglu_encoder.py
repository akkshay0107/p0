import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUEncoderLayer(nn.Module):
    """
    Lean encoder layer for the exact case:
    - batch_first=True
    - norm_first=True
    - dropout=0.0
    - self-attention only
    - no masks
    - packed SwiGLU FFN
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        *,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
        swiglu_hidden: int | None = None,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # pre norm only
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # fused qkv proj
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # dim ff assumed for a gelu ffn (to live with the old impl that uses default torch transformer)
        # 2/3 correction to approx use same parameters
        if swiglu_hidden is None:
            swiglu_hidden = (2 * dim_feedforward) // 3
            swiglu_hidden = (swiglu_hidden + 7) & ~7  # round up to nearest 8
            # for p = 2^k, round up using (x + (p-1)) & ~(p-1) if needed to pad longer

        self.swiglu_hidden = swiglu_hidden

        # packed swiglu proj (both branches together)
        self.w13 = nn.Linear(d_model, 2 * swiglu_hidden, bias=bias)
        self.w2 = nn.Linear(swiglu_hidden, d_model, bias=bias)

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, S, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, S, D
        q, k, v = qkv.unbind(0)  # each become B, H, S, D

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        # B, H, S, D -> B, S, H, D and then combine H, D into d_model
        x = x.transpose(1, 2).reshape(B, S, self.d_model)
        return self.out_proj(x)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        # split last dim of 2 * hidden into seperate streams
        gate, val = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * val)

    def forward(
        self,
        src: torch.Tensor,
    ) -> torch.Tensor:
        x = src + self._self_attention(self.norm1(src))
        x = x + self._ffn(self.norm2(x))
        return x


class SwiGLUTransformerEncoder(nn.Module):
    """
    Near-direct replacement for nn.TransformerEncoder in the specific setup:
    - cloned identical layers
    - no final encoder norm
    - batch_first=True inputs
    - no nested tensor path
    - no masks
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        *,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
        swiglu_hidden: int | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SwiGLUEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    bias=bias,
                    layer_norm_eps=layer_norm_eps,
                    swiglu_hidden=swiglu_hidden,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        src: torch.Tensor,
    ) -> torch.Tensor:
        x = src
        for layer in self.layers:
            x = layer(x)
        return x
