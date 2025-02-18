import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from models.original_wanda_vit.components.PatchEmbedding import PatchEmbedding
from models.original_wanda_vit.components.TransformerBlock import TransformerBlock
from models.original_wanda_vit.utils import init_weights


class VisionTransformer(nn.Module):
    """
    Vision Transformer
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        self.drop_rate = drop_rate
        attn_drop_rate = drop_rate

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.depth = depth

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x[:, 0])

        return x

    def update_drop_path(self, drop_path_rate):
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]

        for i in range(self.depth):
            self.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        self.drop_rate = drop_rate

        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate
