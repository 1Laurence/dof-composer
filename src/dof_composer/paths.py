"""模型权重的自动查找。

存在的意义只有一个：让用户不必手写一长串绝对路径。
程序会从"当前目录"开始逐级往上找，命中几个常见布局就返回。

查找顺序刻意设计成"越近的越优先"，这样同一台机器上有多份权重副本时，
用户放在当前项目里的那份会被采用，而不是被某个远处的旧版本覆盖。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: BiRefNet 权重的候选相对路径。含 ``config.json`` 才认，否则可能是空目录。
BIREFNET_CANDIDATES: tuple[str, ...] = (
    "models/BiRefNet",
    "weights/BiRefNet",
    "checkpoints/BiRefNet",
    "1.ZhengPeng7_BiRefNet/models/BiRefNet",  # 作者本机的小土堆式目录命名
)

#: Depth-Anything-V2 代码仓库的候选相对路径。
DEPTH_REPO_CANDIDATES: tuple[str, ...] = (
    "Depth-Anything-V2",
    "third_party/Depth-Anything-V2",
    "third_party/depth-anything-v2",
    "2.DepthAnything_Depth-Anything-V2",  # 作者本机
)

#: 向上搜索的最大层数。再多就属于"乱翻用户硬盘"了。
MAX_SEARCH_DEPTH = 4


def _iter_search_roots(start: Path) -> list[Path]:
    """从 start 开始逐级往上，列出用于搜索的根目录。"""
    roots = [Path(start).resolve()]
    current = roots[0]
    for _ in range(MAX_SEARCH_DEPTH):
        if current.parent == current:
            break
        current = current.parent
        roots.append(current)
    return roots


def _all_search_starts() -> list[Path]:
    """所有搜索起点，按优先级排序。

    1. 当前工作目录 —— 用户的直觉是"我在哪个目录跑，就找哪个目录"
    2. 本包的安装位置 —— 覆盖"人在别处、代码装在项目里"的情形
       （``pip install -e .`` 时包位置就在项目内，这是最常见的场景）
    """
    starts = [Path.cwd()]
    package_dir = Path(__file__).resolve().parent
    if package_dir not in starts:
        starts.append(package_dir)
    return starts


def _search(starts: list[Path] | None, candidates: tuple[str, ...], probe: str) -> Path | None:
    """在给定的起点里按候选路径逐个探测。

    Args:
        starts: 搜索起点；None 表示用 :func:`_all_search_starts`。
        candidates: 候选相对路径。
        probe: 用来判断"目录真的可用"的探针文件（相对候选路径）。
    """
    for start in starts or _all_search_starts():
        for root in _iter_search_roots(start):
            for relative in candidates:
                candidate = root / relative
                if (candidate / probe).is_file():
                    logger.debug("自动找到：%s", candidate)
                    return candidate
    return None


def find_birefnet_weights(start: Path | None = None) -> Path | None:
    """查找本地 BiRefNet 权重目录。

    Returns:
        含 ``config.json`` 的权重目录，找不到返回 None。
        返回 None 时调用方应回退到 HuggingFace repo id。
    """
    starts = [start] if start is not None else None
    return _search(starts, BIREFNET_CANDIDATES, probe="config.json")


def find_depth_repo(start: Path | None = None) -> Path | None:
    """查找本地 Depth-Anything-V2 代码仓库根目录。"""
    starts = [start] if start is not None else None
    return _search(starts, DEPTH_REPO_CANDIDATES, probe="depth_anything_v2/dpt.py")


def find_depth_checkpoint(
    repo_root: Path | None,
    encoder: str,
    start: Path | None = None,
) -> Path | None:
    """查找深度权重 ``.pth``。

    先在仓库的 ``checkpoints/`` 下找约定文件名，再在几个常见位置兜底。
    """
    filename = f"depth_anything_v2_{encoder}.pth"

    if repo_root is not None:
        conventional = repo_root / "checkpoints" / filename
        if conventional.is_file():
            return conventional

    starts = [start] if start is not None else None
    for origin in starts or _all_search_starts():
        for root in _iter_search_roots(origin):
            for relative in ("checkpoints", "models", "weights", ""):
                candidate = (root / relative / filename) if relative else (root / filename)
                if candidate.is_file():
                    return candidate
    return None
