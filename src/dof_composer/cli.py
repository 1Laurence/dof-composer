"""命令行入口。

用法::

    python -m dof_composer --input photo.jpg --outdir out --da2-root ./Depth-Anything-V2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from dof_composer import __version__
from dof_composer.composer import compose_depth_of_field, compute_sharpness_weight
from dof_composer.config import (
    DEFAULT_BLUR_SIGMA,
    DEFAULT_DEPTH_INPUT_SIZE,
    DEPTH_ENCODERS,
)
from dof_composer.depth_estimator import DepthEstimator
from dof_composer.image_io import (
    collect_image_paths,
    load_bgr_array,
    load_rgb_image,
    make_output_stem,
    save_alpha_png,
    save_bgr_png,
    save_gray_png,
)
from dof_composer.paths import (
    find_birefnet_weights,
    find_depth_checkpoint,
    find_depth_repo,
)
from dof_composer.segmenter import SubjectSegmenter

logger = logging.getLogger(__name__)

#: 输出文件的名称前缀。集中定义，避免各处的字符串散落。
OUTPUT_PREFIX_ALPHA = "alpha"
OUTPUT_PREFIX_DEPTH = "depth"
OUTPUT_PREFIX_DEPTH_COLOR = "depth_colormap"
OUTPUT_PREFIX_COMPOSITE = "dof"


def build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    设计原则：**只有 ``--input`` 是必需的**，其余全部可省略。
    权重位置会自动查找（见 :mod:`dof_composer.paths`），
    找不到时才需要手动指定。
    """
    parser = argparse.ArgumentParser(
        prog="dof-composer",
        description="给一张照片加上大光圈景深效果。",
        epilog=(
            "示例：\n"
            "  dof-composer photo.jpg          # 最简单的用法，输出到 out/\n"
            "  dof-composer ./photos           # 处理整个目录\n"
            "  dof-composer photo.jpg -o result --blur-sigma 20\n"
            "\n"
            "权重会自动在常见位置查找。找不到时用 --birefnet-weights 和 --da2-root 指定。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="输入图片路径，或包含图片的目录",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("out"),
        help="输出目录，不存在会自动创建（默认 out/）",
    )
    parser.add_argument(
        "-w",
        "--birefnet-weights",
        default=None,
        help="BiRefNet 权重目录或 HuggingFace repo id。省略时自动查找",
    )
    parser.add_argument(
        "-d",
        "--da2-root",
        type=Path,
        default=None,
        help="Depth-Anything-V2 代码仓库根目录。省略时自动查找",
    )
    parser.add_argument(
        "-c",
        "--da2-checkpoint",
        type=Path,
        default=None,
        help="深度权重 .pth 路径。省略时自动查找",
    )
    parser.add_argument(
        "-e",
        "--encoder",
        default="vitl",
        choices=sorted(DEPTH_ENCODERS),
        help="深度模型规格，越大约准也越慢（默认 vitl）",
    )
    parser.add_argument(
        "--depth-input-size",
        type=int,
        default=DEFAULT_DEPTH_INPUT_SIZE,
        help="深度推理边长，调大细节更好但更慢",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=DEFAULT_BLUR_SIGMA,
        help="背景高斯模糊强度，越大背景越糊",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="输出详细日志",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def render_depth_colormap(normalized_depth: np.ndarray) -> np.ndarray:
    """把归一化深度渲染成伪彩图，便于肉眼检查深度是否合理。

    返回 BGR 顺序的 uint8 数组。
    """
    import matplotlib

    rgb = matplotlib.cm.Spectral_r(normalized_depth)[:, :, :3]
    # matplotlib 给 RGB，cv2.imwrite 要 BGR
    bgr = rgb[:, :, ::-1]
    # 用 round 不用 astype：astype 是截断，1.0*255 附近的浮点误差会把
    # 最远处压成 254，伪彩图整体偏暗一档，肉眼比对深度范围时会误判。
    return np.round(bgr * 255).astype(np.uint8)


def resolve_weights(args: argparse.Namespace) -> tuple[str, Path, Path]:
    """确定三个权重路径：BiRefNet 目录、Depth 仓库、Depth 权重文件。

    用户显式给了就用用户的，没给就自动查找。
    找不到时抛出带操作指引的错误，而不是让底层抛 ``FileNotFoundError``。

    Returns:
        ``(birefnet_weights, depth_repo_root, depth_checkpoint)``
    """
    # --- BiRefNet ---
    if args.birefnet_weights:
        birefnet_weights: str | Path = args.birefnet_weights
    else:
        found = find_birefnet_weights()
        if found is None:
            # 找不到本地权重时回退到在线拉取，这是最省事的默认行为
            birefnet_weights = "ZhengPeng7/BiRefNet"
            logger.info(
                "本地未找到 BiRefNet 权重，将从 HuggingFace 下载（首次约 445MB）"
            )
        else:
            birefnet_weights = found
            logger.info("BiRefNet 权重：%s", found)

    # --- Depth 仓库 ---
    depth_repo = args.da2_root
    if depth_repo is None:
        depth_repo = find_depth_repo()
        if depth_repo is None:
            raise FileNotFoundError(
                "找不到 Depth-Anything-V2 代码仓库。请先克隆：\n"
                "  git clone https://github.com/DepthAnything/Depth-Anything-V2\n"
                "然后用 -d 指定它的位置，例如：\n"
                "  dof-composer photo.jpg -d ./Depth-Anything-V2"
            )
        logger.info("Depth-Anything-V2 仓库：%s", depth_repo)

    # --- Depth 权重 ---
    depth_checkpoint = args.da2_checkpoint
    if depth_checkpoint is None:
        depth_checkpoint = find_depth_checkpoint(depth_repo, args.encoder)
        if depth_checkpoint is None:
            raise FileNotFoundError(
                f"找不到深度权重 depth_anything_v2_{args.encoder}.pth。\n"
                "请从 https://github.com/DepthAnything/Depth-Anything-V2 的\n"
                "Pre-trained Models 表格下载，放到仓库的 checkpoints/ 目录下：\n"
                f"  {depth_repo / 'checkpoints' / f'depth_anything_v2_{args.encoder}.pth'}"
            )
        logger.info("深度权重：%s", depth_checkpoint)

    return birefnet_weights, depth_repo, depth_checkpoint


def run(args: argparse.Namespace) -> int:
    """执行完整的处理流程。

    Returns:
        进程退出码，0 表示全部成功。
    """
    outdir: Path = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        image_paths = collect_image_paths(args.input)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    try:
        birefnet_weights, depth_repo, depth_checkpoint = resolve_weights(args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("待处理 %d 张图片，输出到 %s", len(image_paths), outdir)

    # ---------------- 第一阶段：BiRefNet 出 alpha ----------------
    segmenter = SubjectSegmenter(birefnet_weights, args.device)
    segmenter.warm_up(load_rgb_image(image_paths[0]))

    alpha_by_stem: dict[str, np.ndarray] = {}
    for image_path in image_paths:
        image = load_rgb_image(image_path)
        alpha = segmenter.predict_alpha(image)
        alpha_by_stem[image_path.stem] = alpha

        alpha_path = outdir / f"{make_output_stem(image_path, OUTPUT_PREFIX_ALPHA)}.png"
        save_alpha_png(alpha, alpha_path)
        logger.info("[%s] alpha 已保存：%s", image_path.name, alpha_path.name)

    # 必须在加载深度模型之前释放，否则 4GB 显存装不下两个模型
    segmenter.release()

    # ---------------- 第二阶段：深度 + 合成 ----------------
    estimator = DepthEstimator(
        encoder=args.encoder,
        checkpoint=depth_checkpoint,
        source_root=depth_repo,
        device=args.device,
    )

    first_bgr = load_bgr_array(image_paths[0])
    if first_bgr is not None:
        estimator.warm_up(first_bgr, args.depth_input_size)

    failures = 0
    for image_path in image_paths:
        image_bgr = load_bgr_array(image_path)
        if image_bgr is None:
            failures += 1
            continue

        raw_depth = estimator.predict_relative_depth(image_bgr, args.depth_input_size)
        normalized_depth = estimator.normalize(raw_depth)

        save_gray_png(
            normalized_depth,
            outdir / f"{make_output_stem(image_path, OUTPUT_PREFIX_DEPTH)}.png",
        )
        save_bgr_png(
            render_depth_colormap(normalized_depth),
            outdir / f"{make_output_stem(image_path, OUTPUT_PREFIX_DEPTH_COLOR)}.png",
        )

        alpha = alpha_by_stem[image_path.stem]
        sharpness_weight = compute_sharpness_weight(alpha, normalized_depth)
        composite = compose_depth_of_field(image_bgr, sharpness_weight, args.blur_sigma)

        save_bgr_png(
            composite,
            outdir / f"{make_output_stem(image_path, OUTPUT_PREFIX_COMPOSITE)}.png",
        )
        logger.info("[%s] 合成完成", image_path.name)

    estimator.release()

    if failures:
        logger.error("%d 张图片处理失败", failures)
        return 1

    logger.info("全部完成，结果在 %s", outdir)
    return 0


def main(argv: list[str] | None = None) -> int:
    """程序入口。"""
    args = build_argument_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    try:
        return run(args)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        # 这些是用户可自行修复的问题，给一行提示即可，不打印 traceback
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
