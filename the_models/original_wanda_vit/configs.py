from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _default_config(url="", **kwargs):
    """
    Returns a default configuration for a Vision Transformer model, as dictionary.
    """

    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


AVAILABLE_CONFIGS = {
    # Patch models
    "vit_small_patch16_224": _default_config(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/"
        "vit_small_p16_224-15ec54c9.pth",
    ),
    "vit_base_patch16_224": _default_config(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/"
        "jx_vit_base_p16_224-80ecf9dd.pth",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_base_patch16_384": _default_config(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/"
        "jx_vit_base_p16_384-83fb41ba.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_base_patch32_384": _default_config(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/"
        "jx_vit_base_p32_384-830016f5.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_large_patch16_224": _default_config(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/"
        "jx_vit_large_p16_224-4ee7a4dc.pth",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch16_384": _default_config(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/"
        "jx_vit_large_p16_384-b3be5167.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_large_patch32_384": _default_config(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/"
        "jx_vit_large_p32_384-9b920ba8.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
}
