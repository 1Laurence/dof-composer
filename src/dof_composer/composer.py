"""景深合成：把 alpha 遮罩与深度图融合成清晰度权重，再混合原图与模糊图。

这是整个项目的核心，也是唯一一处真正的"算法"。
其余模块都在为它准备输入、或保存它的输出。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["compute_sharpness_weight", "compose_depth_of_field"]


def _check_unit_range(name: str, arr: np.ndarray, hint: str = "") -> None:
    """确认数组是 [0, 1] 的浮点数，否则给出能直接照做的报错。

    为什么要有这道检查：``cv2.imread(path, cv2.IMREAD_GRAYSCALE)`` 返回的是
    **uint8、值域 0~255**。直接丢进 ``alpha + (1-alpha)*depth`` 时，
    ``alpha=255`` 会让 ``1-alpha = -254``，权重瞬间变成 -6 万这种量级，
    下游混合出来的是一张高通残差图（边缘全是彩色噪点），而不是景深合成。

    这种"静默算错"比直接报错恶劣得多 —— 调用方会先怀疑算法、怀疑参数、
    怀疑模型，最后才怀疑到读图方式上。所以入口就把话说清楚。
    """
    if arr.dtype == np.uint8:
        raise ValueError(
            f"{name} 是 uint8（值域 0~255），但这里需要 float32 的 [0, 1]。\n"
            f"{hint}"
            f"正确写法：\n"
            f"    {name} = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0\n"
            f"如果手上是 PNG/TIF 的 8 位灰度图，除以 255 就够了；"
            f"若是 0~65535 的 16 位图，请除以 65535。"
        )

    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(
            f"{name} 的 dtype 是 {arr.dtype}，需要浮点类型（推荐 float32）。{hint}"
        )

    lo, hi = float(arr.min()), float(arr.max())
    if lo < -1e-6 or hi > 1.0 + 1e-6:
        raise ValueError(
            f"{name} 的取值超出 [0, 1]：min={lo:.4f}, max={hi:.4f}。{hint}"
            f"如果值域是 0~255，请先 /255.0；如果是深度模型的原始输出，"
            f"请先做 min-max 归一化。"
        )


def compute_sharpness_weight(
    alpha: np.ndarray,
    normalized_depth: np.ndarray,
) -> np.ndarray:
    """融合 alpha 与深度，得到每个像素"应该有多清晰"的权重。

    公式::

        weight = alpha + (1 - alpha) * depth

    两个信号各回答一个问题：

    - ``alpha`` 回答"这**是不是**主体"（1 = 确定是，0 = 确定不是）
    - ``depth`` 回答"这**有多远**"（1 = 最近，0 = 最远）

    为什么是这个形式，而不是简单相加或相乘：

    - ``alpha = 1`` → ``weight = 1``。确定为主体时强制清晰，
      深度信息被完全屏蔽。这是刻意的：深度模型偶尔会把主体误判成远处，
      若不屏蔽会导致主体边缘被糊掉。
    - ``alpha = 0`` → ``weight = depth``。确定是背景时，虚化程度完全交给深度，
      远处糊得更狠，近处糊得少，形成纵深。
    - ``alpha`` 在 0~1 之间（软边）→ 按 alpha 在两者之间插值，边缘过渡自然。

    Args:
        alpha: ``(H, W)`` float32，值域 [0, 1]。
        normalized_depth: ``(H, W)`` float32，值域 [0, 1]，已 min-max 归一化。

    Returns:
        ``(H, W)`` float32，值域 [0, 1]，1 表示保持清晰。
    """
    if alpha.shape != normalized_depth.shape:
        raise ValueError(
            f"alpha 形状 {alpha.shape} 与深度图 {normalized_depth.shape} 不一致，"
            "无法逐像素运算。"
        )

    _check_unit_range("alpha", alpha)
    _check_unit_range("normalized_depth", normalized_depth)

    return alpha + (1.0 - alpha) * normalized_depth


def compose_depth_of_field(
    image_bgr: np.ndarray,
    sharpness_weight: np.ndarray,
    blur_sigma: float,
) -> np.ndarray:
    """按清晰度权重混合原图与高斯模糊图。

    Args:
        image_bgr: ``(H, W, 3)`` uint8，BGR 顺序的原始图像。
        sharpness_weight: ``(H, W)`` float32，值域 [0, 1]，来自
            :func:`compute_sharpness_weight`。
        blur_sigma: 高斯模糊的标准差。核大小传 ``(0, 0)``，
            让 OpenCV 按 sigma 自动推算（约 ``6*sigma+1``，取奇数）。

    Returns:
        ``(H, W, 3)`` uint8，BGR 顺序的合成结果。

    Note:
        ``sharpness_weight[..., None]`` 是把 ``(H, W)`` 扩成 ``(H, W, 1)``，
        这样才能和三通道的 ``(H, W, 3)`` 图广播相乘。
        少了这一步会报 ``operands could not be broadcast together``。
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"期望 (H, W, 3) 的 BGR 图像，实际形状 {image_bgr.shape}")

    _check_unit_range("sharpness_weight", sharpness_weight)

    if sharpness_weight.shape[:2] != image_bgr.shape[:2]:
        raise ValueError(
            f"权重图尺寸 {sharpness_weight.shape[:2]} 与原图 "
            f"{image_bgr.shape[:2]} 不一致。二者必须逐像素对应。"
        )

    blurred = cv2.GaussianBlur(image_bgr, (0, 0), blur_sigma)

    weight_3ch = sharpness_weight[..., None]
    composite = image_bgr.astype(np.float32) * weight_3ch + blurred.astype(
        np.float32
    ) * (1.0 - weight_3ch)

    # 转 uint8 前必须 clip：浮点运算可能溢出 [0, 255]，
    # 直接 astype(np.uint8) 会按模 256 回绕，比如 256 → 0、-1 → 255。
    return np.clip(composite, 0, 255).astype(np.uint8)
