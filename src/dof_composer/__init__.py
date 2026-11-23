"""dof-composer：用单目深度与主体分割给全焦照片合成假景深。

三层用法，按需要选：

**1. 命令行（最简单）**::

    dof-composer photo.jpg

**2. 只借用合成公式**（不需要模型，纯 numpy）::

    from dof_composer import compute_sharpness_weight, compose_depth_of_field

    weight = compute_sharpness_weight(alpha, normalized_depth)
    result = compose_depth_of_field(image_bgr, weight, blur_sigma=12.0)

**3. 自己组装流程**（可替换其中任一模型）::

    from dof_composer import SubjectSegmenter, DepthEstimator

    seg = SubjectSegmenter("ZhengPeng7/BiRefNet", device="cuda")
    alpha = seg.predict_alpha(pil_image)
    seg.release()

    est = DepthEstimator(encoder="vitl", checkpoint=ckpt,
                         source_root=repo, device="cuda")
    depth = est.normalize(est.predict_relative_depth(bgr, input_size=518))

    这两个类只负责"出 alpha"和"出深度"，怎么合成由你自己决定。
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "compose_depth_of_field",
    "compute_sharpness_weight",
    "DepthEstimator",
    "SubjectSegmenter",
]

#: 延迟导出的符号 → (模块路径, 符号名)
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "compose_depth_of_field": ("dof_composer.composer", "compose_depth_of_field"),
    "compute_sharpness_weight": ("dof_composer.composer", "compute_sharpness_weight"),
    "SubjectSegmenter": ("dof_composer.segmenter", "SubjectSegmenter"),
    "DepthEstimator": ("dof_composer.depth_estimator", "DepthEstimator"),
}

#: 常见缺失依赖 → 可直接照做的安装命令。
#: 键必须与 ``ModuleNotFoundError.name`` 完全一致，避免把
#: ``torch._C`` 这类内部模块缺失误报成"没装 torch"。
_INSTALL_HINTS: dict[str, str] = {
    "torch": (
        "torch 的 wheel 绑定了具体 CUDA 版本，本项目没有替你装。按自己的环境选一个：\n"
        "    CUDA 12.6 → pip install torch torchvision --index-url "
        "https://download.pytorch.org/whl/cu126\n"
        "    纯 CPU    → pip install torch torchvision --index-url "
        "https://download.pytorch.org/whl/cpu"
    ),
    "torchvision": "pip install torchvision（版本需与 torch 对应）",
    "transformers": "pip install \"transformers>=4.39\"",
    "PIL": "pip install Pillow",
    "cv2": "pip install opencv-python",
    "numpy": "pip install numpy",
    "matplotlib": "pip install matplotlib",
}


def _import_symbol(module_path: str, symbol: str, name: str):
    """导入指定符号，缺依赖时把裸的 ModuleNotFoundError 换成能照做的提示。

    为什么值得这么做：用户 ``from dof_composer import SubjectSegmenter`` 时
    拿到的是 ``ModuleNotFoundError: No module named 'torch'`` ——
    既没说这是哪个功能需要它，也没给安装命令，而 torch 的安装又偏偏
    是最容易装错的那个（CPU/CUDA 版本不匹配）。

    注意只对**顶层包本身缺失**给提示。像 ``torch._C`` 缺失说明 torch 装了但
    坏了，那是另一种问题，原样抛出更有诊断价值。
    """
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        hint = _INSTALL_HINTS.get(exc.name or "")
        if hint is None:
            raise
        raise ImportError(
            f"无法导入 {name}：缺少 {exc.name}。\n"
            f"{hint}\n"
            f"（只借合成公式不需要它："
            f"from dof_composer import compute_sharpness_weight, compose_depth_of_field）"
        ) from exc

    return getattr(module, symbol)


def __getattr__(name: str):
    """延迟导出。

    不在 ``__init__`` 里直接 import，是为了让 ``import dof_composer`` 不强制
    加载 torch / cv2。只用合成公式或只读版本号的场景不必付出这个代价。
    """
    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, symbol = export
    return _import_symbol(module_path, symbol, name)


def __dir__() -> list[str]:
    """让 ``dir(dof_composer)`` 只列出公开 API。

    没有这个，``dir()`` 只能看到模块 ``__dict__`` 里的东西：要么一个符号都没有，
    要么在导入过一次之后冒出 ``segmenter`` / ``depth_estimator`` 这些**内部子模块名**，
    再混进 ``annotations``（``from __future__`` 的产物）和 ``importlib``。
    后者更容易误导 —— 用户会以为 ``dof_composer.segmenter`` 是公开 API，
    但它其实是实现细节，随时可能改名。

    因此这里**不**基于 ``globals()`` 推导，直接返回 ``__all__``。
    """
    return sorted(__all__)
