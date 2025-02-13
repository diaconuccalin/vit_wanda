from timm.layers import DropPath
from torch import nn

from models.original_wanda.components.MultiHeadSelfAttentionWithProjectionAndDropout import (
    MultiHeadSelfAttentionWithProjectionAndDropout,
)
from models.original_wanda.components.PositionWiseFeedForwardWithDropout import (
    PositionWiseFeedForwardWithDropout,
)


class TransformerBlock(nn.Module):
    """
    The transformer block.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttentionWithProjectionAndDropout(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = PositionWiseFeedForwardWithDropout(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
