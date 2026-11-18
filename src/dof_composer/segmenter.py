"""BiRefNet 主体分割封装。

职责单一：给一张 RGB 图，返回与原图同尺寸的 float32 alpha 遮罩（值域 [0, 1]）。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from dof_composer.config import BIREFNET_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


class SubjectSegmenter:
    """基于 BiRefNet 的二分图像分割。

    该类只负责"哪些像素属于主体"，不关心深度、景深等下游概念。

    Args:
        weights: HuggingFace repo id（如 ``ZhengPeng7/BiRefNet``）或本地权重目录。
        device: ``cuda`` / ``cpu`` / ``cuda:1`` 等。
    """

    def __init__(self, weights: str | Path, device: str) -> None:
        # 延迟导入：没装 transformers 的人只跑构图部分时不该被它拦住
        from transformers import AutoModelForImageSegmentation

        self.device = torch.device(device)
        self._input_size = BIREFNET_INPUT_SIZE

        resolved = self._resolve_weights(weights)
        logger.info("加载 BiRefNet 权重：%s", resolved)
        # trust_remote_code 是必需的：BiRefNet 的模型类是随权重下发的自定义代码
        self._model = (
            AutoModelForImageSegmentation.from_pretrained(
                resolved,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            .eval()
            .to(self.device)
        )

        self._preprocess = transforms.Compose(
            [
                transforms.Resize((self._input_size, self._input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
            ]
        )

    @staticmethod
    def _resolve_weights(weights: str | Path) -> str:
        """把用户给的权重位置规范成 ``from_pretrained`` 能接受的形式。

        ``transformers`` 对参数的判断是二选一的：

        - 如果看起来像 HuggingFace repo id（``namespace/name``），就去线上下载；
        - 否则当成**相对当前工作目录**的本地路径处理。

        问题在于 ``../models/BiRefNet`` 这种以点开头的相对路径会被误判成 repo id，
        抛出 ``OSError: Repo id must be in the form 'repo_name'``。
        因此这里主动把本地路径转成绝对路径，消除歧义。

        Returns:
            绝对路径字符串（本地权重），或原样的 repo id。
        """
        candidate = Path(str(weights)).expanduser()

        # 判断"是否指向一个真实存在的本地目录"，是就转绝对路径
        if candidate.is_dir():
            return str(candidate.resolve())

        # 以 . 或 / 或盘符开头，说明用户明确想指本地路径，但目录不存在
        raw = str(weights)
        looks_like_path = raw.startswith((".", "/", "\\", "~")) or (
            len(raw) > 1 and raw[1] == ":"
        )
        if looks_like_path:
            raise FileNotFoundError(
                f"BiRefNet 权重目录不存在：{candidate.resolve()}\n"
                "请确认路径正确，或改用 HuggingFace repo id（如 ZhengPeng7/BiRefNet）。"
            )

        # 其余情况当作 HuggingFace repo id，交给 transformers 去下载
        return raw

    def warm_up(self, sample: Image.Image) -> None:
        """用一张图空跑一次。

        首次前向包含 CUDA kernel 编译与显存分配，若计入耗时统计会严重偏高，
        因此正式计时前先预热一次，顺便清空缓存、重置峰值统计。
        """
        tensor = self._preprocess(sample).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self._model(tensor)
        self._sync()

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        """预测主体遮罩。

        Args:
            image: RGB 图像，尺寸任意。

        Returns:
            ``(H, W)`` 的 float32 数组，值域 [0, 1]，H/W 与原图一致。
        """
        height, width = image.size[1], image.size[0]

        input_tensor = self._preprocess(image).unsqueeze(0).to(self.device)

        started_at = time.perf_counter()
        with torch.no_grad():
            # 模型返回 list 而非 tensor。最后一个元素是全分辨率预测，
            # 前面几个是训练期使用的低分辨率辅助输出，推理时不需要。
            raw_output = self._model(input_tensor)[-1]
            self._sync()
        logger.info("BiRefNet 前向耗时 %.3f 秒", time.perf_counter() - started_at)

        if self.device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            logger.info("显存峰值 %.2f GB", peak_gb)

        probabilities = raw_output.sigmoid().cpu().float()

        # 模型输出固定在 1024x1024，需还原到原图尺寸。
        # 注意：PIL 的 size 是 (宽, 高)，而 transforms.Resize 要 (高, 宽)。
        resized = transforms.Resize((height, width))(probabilities)

        # (1, 1, H, W) → (H, W)：去掉 batch 维与单通道维
        return resized[0, 0].numpy()

    def release(self) -> None:
        """卸载模型并释放显存。

        BiRefNet 与深度模型同时驻留会撑爆 4GB 显卡，
        因此必须在本类用完、加载下一个模型之前调用。
        """
        del self._model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info(
                "BiRefNet 已卸载，当前显存占用 %.2f GB",
                torch.cuda.memory_allocated() / 1024**3,
            )

    def _sync(self) -> None:
        """等待 GPU 队列真正执行完毕。

        不调用它的话，``time`` 测到的只是"任务提交完成"的时间，不是计算时间。
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize()
