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


def _repair_path(raw: str) -> str | None:
    """把路径里的控制字符反推回它原本该是的样子。

    ``\\2`` 被求值成码位 2 之后，原始文本已经不可知了，
    但常见来源只有两种：八进制转义（``\\1``~``\\7``）与具名转义（``\\n``/``\\t``/``\\b``…）。
    逐个还原成候选，谁真实存在就用谁。全都不存在说明猜错了，返回 None。
    """
    named = {
        7: "\\a",
        8: "\\b",
        9: "\\t",
        10: "\\n",
        11: "\\v",
        12: "\\f",
        13: "\\r",
    }

    def octal_variant(text: str) -> str:
        out = []
        for char in text:
            code = ord(char)
            out.append(f"\\{code}" if 0 <= code <= 7 else char)
        return "".join(out)

    def named_variant(text: str) -> str:
        out = []
        for char in text:
            code = ord(char)
            if code < 32 and code in named:
                out.append(named[code])
            elif 0 <= code <= 7:
                out.append(f"\\{code}")
            else:
                out.append(char)
        return "".join(out)

    for variant in (octal_variant(raw), named_variant(raw)):
        if Path(variant).exists():
            return variant
    return None


def describe_control_chars(raw: str) -> str | None:
    """检查字符串里是否混入了控制字符，有就返回一句能看懂的诊断。

    为什么需要这个：Windows 路径里 ``study\\2.DepthAnything`` 中的 ``\\2``
    是**合法的八进制转义**，Python 会把它变成码位 2 的控制字符，既不报错
    也不警告（反倒是非法的 ``\\V`` 会触发 SyntaxWarning，等于提醒了你）。
    结果就是路径悄悄少了一个字符，``is_dir()`` 返回 False，
    而报错信息里的路径看起来"跟写的完全一样" —— 因为控制字符不显示。

    同样的坑还有 ``\\1``~``\\7``、``\\n``、``\\t``、``\\b``、``\\f`` 等。
    用户目录以数字开头（如 ``2.DepthAnything``）时几乎必踩。

    Args:
        raw: 用户原样传入的路径字符串。

    Returns:
        有问题时返回多行诊断文本，没问题返回 None。
    """
    bad = [(index, char) for index, char in enumerate(raw) if ord(char) < 32]
    if not bad:
        return None

    index, char = bad[0]
    details = f"第 {index} 个字符是 \\x{ord(char):02x}（不可见控制字符）"
    if len(bad) > 1:
        details += f"，另有 {len(bad) - 1} 处同类问题"

    repaired = _repair_path(raw)
    if repaired is not None:
        # 能反推出真实存在的路径，直接把可粘贴的写法给出去，
        # 而不是让用户对着一条被污染的字符串自己数反斜杠。
        hint = (
            f"你要的路径应该是这个（已确认存在，直接复制）：\n"
            f'    Path(r"{repaired}")\n'
        )
    else:
        hint = (
            "改法（任选其一）：\n"
            '    Path(r"你的路径")              # 加 r 前缀，最省事\n'
            '    Path("F:/your/path")          # 换成正斜杠，Windows 也认\n'
            '    Path("F:\\\\your\\\\path")        # 每个反斜杠写两遍\n'
        )

    return (
        "这个路径里混进了控制字符，所以它指向的位置并不存在。\n"
        f"  {details}\n"
        "  而控制字符不占显示宽度，报错里的路径看着是对的 —— 这就是它难查的原因。\n"
        "根因：Windows 路径里的反斜杠被 Python 当成了转义符。\n"
        "  \\1~\\7 是八进制转义，\\n \\t \\b \\f 也是合法转义，全都不报错。\n"
        f"{hint}"
    )


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
