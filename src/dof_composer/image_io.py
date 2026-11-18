"""图像读写与路径收集。

刻意与算法解耦：这里的函数不知道 alpha、深度、景深这些东西的存在。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dof_composer.config import SUPPORTED_IMAGE_SUFFIXES

logger = logging.getLogger(__name__)


def collect_image_paths(source: Path) -> list[Path]:
    """把用户给的输入整理成待处理的图片列表。

    Args:
        source: 单张图片的路径，或一个目录。目录下按文件名排序取所有支持的图片。

    Returns:
        绝对路径列表，保证非空。

    Raises:
        FileNotFoundError: 路径不存在。
        ValueError: 目录里没有任何支持的图片。
    """
    source = source.expanduser().resolve()

    if source.is_file():
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{source}")

    image_paths = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        supported = "、".join(SUPPORTED_IMAGE_SUFFIXES)
        raise ValueError(f"目录内没有可用图片（支持的格式：{supported}）：{source}")

    return image_paths


def load_rgb_image(path: Path) -> Image.Image:
    """按 RGB 三通道读取图片，丢弃 alpha 通道。

    四通道 PNG 直接送进模型会因为通道数不匹配报错，所以统一转换。
    """
    with Image.open(path) as opened:
        return opened.convert("RGB")


def load_bgr_array(path: Path) -> np.ndarray | None:
    """以 OpenCV 惯例（BGR）读取图片为 uint8 数组。读失败返回 None。"""
    array = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if array is None:
        logger.warning("无法读取图片：%s", path)
    return array


def save_alpha_png(alpha: np.ndarray, path: Path) -> None:
    """把 [0, 1] 的浮点 alpha 存成 8bit 灰度 PNG。

    8bit 只有 256 级，用来看效果足够；若要参与后续精算应存 .npy。
    """
    scaled = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(scaled, mode="L").save(path)


def save_gray_png(array: np.ndarray, path: Path) -> None:
    """把 [0, 1] 的浮点灰度图存成 8bit PNG。"""
    scaled = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), scaled)


def save_bgr_png(array: np.ndarray, path: Path) -> None:
    """存 8bit BGR 图。主交付一律 PNG，避免 JPEG 压缩吃掉细节。"""
    cv2.imwrite(str(path), array)


def make_output_stem(image_path: Path, prefix: str) -> str:
    """拼输出文件名（不含扩展名），形如 ``alpha_img_1``。"""
    return f"{prefix}_{image_path.stem}"
