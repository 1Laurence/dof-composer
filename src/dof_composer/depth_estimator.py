"""Depth-Anything-V2 深度估计封装。

职责单一：给一张 BGR 图，返回归一化到 [0, 1] 的相对深度图。

注意 Depth-Anything-V2 并未发布到 PyPI，需要用户提供代码仓库路径，
因此这里用延迟导入 + 显式报错的方式处理，而不是在模块顶层 import。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dof_composer.config import DEPTH_ENCODERS, EPSILON

logger = logging.getLogger(__name__)


class DepthEstimator:
    """基于 Depth-Anything-V2 的单目相对深度估计。

    Args:
        encoder: 规格名，取值见 :data:`dof_composer.config.DEPTH_ENCODERS`。
        checkpoint: 权重 ``.pth`` 路径。
        source_root: Depth-Anything-V2 代码仓库根目录（含 ``depth_anything_v2/`` 包）。
            若该包已可导入，可传 None。
        device: 计算设备。
    """

    def __init__(
        self,
        encoder: str,
        checkpoint: Path,
        source_root: Path | None,
        device: str,
    ) -> None:
        if encoder not in DEPTH_ENCODERS:
            available = "、".join(DEPTH_ENCODERS)
            raise ValueError(f"未知的 encoder：{encoder}（可选：{available}）")

        spec = DEPTH_ENCODERS[encoder]
        if not spec.commercial_use:
            logger.warning(
                "%s 的权重为 CC-BY-NC-4.0，仅限非商业用途。商用请改用 vits 规格。",
                encoder,
            )

        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"找不到深度权重：{checkpoint}\n"
                "请从 Depth-Anything-V2 的 README 下载对应 .pth 放到 checkpoints/ 下。"
            )

        self.device = torch.device(device)
        depth_anything_class = self._import_depth_anything(source_root)

        logger.info("加载深度权重：%s（%s）", checkpoint, encoder)
        self._model = depth_anything_class(
            encoder=spec.name,
            features=spec.features,
            out_channels=list(spec.out_channels),
        )
        # load_state_dict 是原地加载，返回的是加载报告而不是模型。
        # 写成 `model = model.load_state_dict(...)` 会把模型覆盖成报告对象。
        state_dict = torch.load(checkpoint, map_location="cpu")
        self._model.load_state_dict(state_dict)
        self._model = self._model.eval().to(self.device)

    def warm_up(self, sample_bgr: np.ndarray, input_size: int) -> None:
        """预热一次，理由同 :class:`SubjectSegmenter`。"""
        self.predict_relative_depth(sample_bgr, input_size)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def predict_relative_depth(self, image_bgr: np.ndarray, input_size: int) -> np.ndarray:
        """预测相对深度。

        Args:
            image_bgr: OpenCV 惯例的 BGR uint8 图像。Depth-Anything 内部按 BGR 处理。
            input_size: 推理边长。

        Returns:
            ``(H, W)`` 的 float32 原始深度（未归一化）。
            **这是相对深度，不是物理距离**：只有大小顺序有意义，没有绝对尺度。
        """
        started_at = time.perf_counter()
        with torch.no_grad():
            raw_depth = self._model.infer_image(image_bgr, input_size=input_size)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        logger.info("Depth-Anything 前向耗时 %.3f 秒", time.perf_counter() - started_at)

        if self.device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            logger.info("显存峰值 %.2f GB", peak_gb)

        return raw_depth

    @staticmethod
    def normalize(raw_depth: np.ndarray) -> np.ndarray:
        """min-max 归一化到 [0, 1]。

        Depth-Anything 输出的是逆深度，数值越大越近。
        因此归一化后 ``1.0`` 表示最近，``0.0`` 表示最远。
        """
        depth_min = float(raw_depth.min())
        depth_max = float(raw_depth.max())
        return (raw_depth - depth_min) / (depth_max - depth_min + EPSILON)

    def release(self) -> None:
        """卸载模型并释放显存。"""
        del self._model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def resolve_checkpoint(encoder: str, source_root: Path | None) -> Path:
        """按约定推断权重路径：``<source_root>/checkpoints/depth_anything_v2_<encoder>.pth``。"""
        if source_root is None:
            raise ValueError(
                "未指定 --da2-ckpt 时必须提供 --da2-root，否则无法推断权重位置。"
            )
        return Path(source_root) / "checkpoints" / f"depth_anything_v2_{encoder}.pth"

    @staticmethod
    def _import_depth_anything(source_root: Path | None):
        """导入 DepthAnythingV2 类，必要时把用户给的仓库根目录塞进 sys.path。"""
        try:
            from depth_anything_v2.dpt import DepthAnythingV2

            return DepthAnythingV2
        except ImportError:
            if source_root is None:
                raise ImportError(
                    "找不到 depth_anything_v2 包。\n"
                    "  git clone https://github.com/DepthAnything/Depth-Anything-V2\n"
                    "  然后加参数 --da2-root /path/to/Depth-Anything-V2"
                ) from None

            resolved = Path(source_root).expanduser().resolve()
            package_dir = resolved / "depth_anything_v2"
            if not package_dir.is_dir():
                raise ImportError(
                    f"{resolved} 下没有 depth_anything_v2/ 目录，请确认 --da2-root 指向仓库根目录。"
                ) from None

            # insert(0) 而非 append：确保优先用用户指定的这份代码
            sys.path.insert(0, str(resolved))
            from depth_anything_v2.dpt import DepthAnythingV2

            return DepthAnythingV2
