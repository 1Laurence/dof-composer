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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "compose_depth_of_field",
    "compute_sharpness_weight",
    "DepthEstimator",
    "SubjectSegmenter",
]


def __getattr__(name: str):
    """延迟导出。

    不在 ``__init__`` 里直接 import，是为了让 ``import dof_composer`` 不强制
    加载 torch / cv2。只用合成公式或只读版本号的场景不必付出这个代价。
    """
    if name == "compose_depth_of_field":
        from dof_composer.composer import compose_depth_of_field

        return compose_depth_of_field

    if name == "compute_sharpness_weight":
        from dof_composer.composer import compute_sharpness_weight

        return compute_sharpness_weight

    if name == "SubjectSegmenter":
        from dof_composer.segmenter import SubjectSegmenter

        return SubjectSegmenter

    if name == "DepthEstimator":
        from dof_composer.depth_estimator import DepthEstimator

        return DepthEstimator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
