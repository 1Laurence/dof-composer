"""示例：用 dof-composer 的模型，但换成自己的合成规则。

默认流程是"按深度连续虚化整个背景"。这个例子改成更接近真实镜头的行为：

    - 只把**最远**的一段背景虚化（模拟景深范围之外才失焦）
    - 中间距离保持清晰（模拟景深范围内）
    - 主体无论如何都清晰

用法::

    python custom_compose.py photo.jpg

需要先按 README 装好 dof-composer 和两个模型权重。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dof_composer import DepthEstimator, SubjectSegmenter
from dof_composer.paths import (
    find_birefnet_weights,
    find_depth_checkpoint,
    find_depth_repo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自定义合成规则示例")
    parser.add_argument("input", type=Path, help="输入图片")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("out"))
    parser.add_argument(
        "--encoder", default="vitl", help="深度模型规格（默认 vitl）"
    )
    parser.add_argument(
        "--far-ratio",
        type=float,
        default=0.35,
        help="虚化最远的这么多比例（0~1，默认 0.35）。"
        "内部按深度的分位数取阈值，所以这个数字就是实际被虚化的面积占比。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="直接指定深度阈值（低于此值算'太远'）。"
        "指定后 --far-ratio 失效。注意这是深度的**值**不是分位数，"
        "同一个数字在不同图片上对应的面积可能差很远。",
    )
    parser.add_argument(
        "--feather",
        type=float,
        default=0.15,
        help="过渡带宽度（默认 0.15）。0 会得到硬边，看得出一条'清晰/模糊'分界线。",
    )
    parser.add_argument("--blur-sigma", type=float, default=9.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(args.input))
    if bgr is None:
        print(f"读不了这张图：{args.input}", file=sys.stderr)
        return 1

    pil = Image.open(args.input).convert("RGB")

    # ---------------- 定位权重 ----------------
    # 复用 CLI 的自动查找逻辑，避免在示例里又写一遍路径处理
    birefnet_weights = find_birefnet_weights()
    if birefnet_weights is None:
        # 本地没有就回退到在线拉取（需要能访问 HuggingFace）
        birefnet_weights = "ZhengPeng7/BiRefNet"
        print("本地未找到 BiRefNet 权重，将从 HuggingFace 下载")

    repo = find_depth_repo()
    if repo is None:
        print(
            "找不到 Depth-Anything-V2 仓库。请先克隆：\n"
            "  git clone https://github.com/DepthAnything/Depth-Anything-V2",
            file=sys.stderr,
        )
        return 1

    checkpoint = find_depth_checkpoint(repo, args.encoder)
    if checkpoint is None:
        print(
            f"找不到权重 depth_anything_v2_{args.encoder}.pth，"
            f"应放在 {repo / 'checkpoints'}",
            file=sys.stderr,
        )
        return 1

    # ---------------- 抠图 ----------------
    seg = SubjectSegmenter(birefnet_weights, device=args.device)
    alpha = seg.predict_alpha(pil)
    seg.release()  # 必须卸载，否则 4GB 卡装不下下一个模型

    # ---------------- 深度 ----------------
    est = DepthEstimator(
        encoder=args.encoder,
        checkpoint=checkpoint,
        source_root=repo,
        device=args.device,
    )
    depth = est.normalize(est.predict_relative_depth(bgr, input_size=518))
    est.release()

    # ---------------- 自定义合成 ----------------
    # 默认公式是 weight = alpha + (1-alpha)*depth，深度是连续参与的。
    # 这里改成"只有远到某个程度才糊"：中间景深范围保持清晰，
    # 更接近真实光学的景深概念。
    #
    # 两个容易写错的地方：
    #
    # 1) ``depth < 0.35`` 并不等于"最远的 35%"。depth 是归一化后的**值**，
    #    不是**分位数**，同一个阈值在不同图片上覆盖的面积可能差一倍
    #    （实测某张图 depth<0.35 覆盖 60.6%）。所以这里用分位数取阈值。
    # 2) 直接二值化会在"清晰/模糊"之间留下一条能看出来的硬边。
    #    用 feather 把跳变摊开成渐变，视觉上才像真的景深。
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = float(np.quantile(depth, args.far_ratio))

    if args.feather > 0:
        # depth 越小于 threshold，far 越接近 1（该糊）
        far = np.clip((threshold - depth) / args.feather, 0.0, 1.0)
    else:
        far = (depth < threshold).astype(np.float32)

    far = far.astype(np.float32)
    weight = np.maximum(1.0 - far, alpha)  # alpha 拥有一票否决权

    blurred = cv2.GaussianBlur(bgr, (0, 0), args.blur_sigma)
    composite = bgr.astype(np.float32) * weight[..., None] + blurred.astype(
        np.float32
    ) * (1.0 - weight[..., None])
    # 用 round 而不是直接 astype：astype 是截断，254.9 会变成 254，
    # 大片平坦区域会系统性偏暗一个灰阶。
    result = np.round(np.clip(composite, 0, 255)).astype(np.uint8)

    output_path = args.outdir / f"custom_{args.input.stem}.png"
    cv2.imwrite(str(output_path), result)

    blurred_ratio = 100.0 * (far > 0.5).mean()
    print(f"已保存：{output_path}")
    print(f"虚化阈值 {threshold:.4f}，被虚化区域占全图 {blurred_ratio:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
