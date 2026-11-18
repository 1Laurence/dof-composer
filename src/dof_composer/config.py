"""全局常量与模型规格注册表。

所有"魔法数字"集中在这里，避免散落在业务代码中。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------- 输入尺寸

#: BiRefNet 的固定输入边长。权重按 1024x1024 训练，改动会显著掉精度。
BIREFNET_INPUT_SIZE: Final[int] = 1024

#: Depth-Anything-V2 的默认推理边长。调大细节更好但耗时线性增长。
DEFAULT_DEPTH_INPUT_SIZE: Final[int] = 518

# ---------------------------------------------------------------- 合成参数

#: 背景高斯模糊的标准差。越大背景越糊。
DEFAULT_BLUR_SIGMA: Final[float] = 12.0

#: ImageNet 统计量。两个模型都按这套分布做归一化，不可随意改动。
IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

#: 归一化时防止除零的微小量。
EPSILON: Final[float] = 1e-6

# ---------------------------------------------------------------- 模型规格


@dataclass(frozen=True)
class DepthEncoderSpec:
    """Depth-Anything-V2 的一种规格。

    ``features`` 与 ``out_channels`` 必须和权重完全对应，
    否则 ``load_state_dict`` 会抛 ``size mismatch``。
    """

    name: str
    features: int
    out_channels: tuple[int, ...]
    commercial_use: bool
    note: str


DEPTH_ENCODERS: Final[dict[str, DepthEncoderSpec]] = {
    "vits": DepthEncoderSpec(
        name="vits",
        features=64,
        out_channels=(48, 96, 192, 384),
        commercial_use=True,
        note="Small，Apache-2.0，可商用",
    ),
    "vitb": DepthEncoderSpec(
        name="vitb",
        features=128,
        out_channels=(96, 192, 384, 768),
        commercial_use=False,
        note="Base，CC-BY-NC-4.0，禁止商用",
    ),
    "vitl": DepthEncoderSpec(
        name="vitl",
        features=256,
        out_channels=(256, 512, 1024, 1024),
        commercial_use=False,
        note="Large，CC-BY-NC-4.0，禁止商用",
    ),
    "vitg": DepthEncoderSpec(
        name="vitg",
        features=384,
        out_channels=(1536, 1536, 1536, 1536),
        commercial_use=False,
        note="Giant，CC-BY-NC-4.0，禁止商用",
    ),
}

#: 可接受的图片扩展名（小写，含点）。
SUPPORTED_IMAGE_SUFFIXES: Final[tuple[str, ...]] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
)
