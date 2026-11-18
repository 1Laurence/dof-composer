"""合成公式的单元测试。

这些测试不加载任何模型，纯 numpy 运算，秒级完成。
它们锁住的是公式的**语义契约**，将来重构时若改坏了会立刻失败。
"""

from __future__ import annotations

import numpy as np
import pytest

from dof_composer.composer import compose_depth_of_field, compute_sharpness_weight


class TestComputeSharpnessWeight:
    """验证 ``weight = alpha + (1 - alpha) * depth`` 的边界行为。"""

    def test_subject_region_is_forced_sharp(self):
        """alpha=1 时必须完全清晰，深度值再怎么变都不能影响结果。"""
        alpha = np.ones((4, 4), dtype=np.float32)
        depth = np.array([[0.0, 0.3, 0.5, 1.0]] * 4, dtype=np.float32)

        weight = compute_sharpness_weight(alpha, depth)

        # 主体保护优先：即使深度说"这里很远"，也必须保持 w=1
        np.testing.assert_allclose(weight, 1.0, atol=1e-6)

    def test_background_follows_depth(self):
        """alpha=0 时权重应完全等于深度值。"""
        alpha = np.zeros((4, 4), dtype=np.float32)
        depth = np.array([[0.0, 0.25, 0.75, 1.0]] * 4, dtype=np.float32)

        weight = compute_sharpness_weight(alpha, depth)

        np.testing.assert_allclose(weight, depth, atol=1e-6)

    def test_soft_edge_interpolates(self):
        """alpha=0.5 时应取两者的中点。"""
        alpha = np.full((2, 2), 0.5, dtype=np.float32)
        depth = np.ones((2, 2), dtype=np.float32)

        weight = compute_sharpness_weight(alpha, depth)

        # 0.5 + 0.5 * 1.0 = 1.0
        np.testing.assert_allclose(weight, 1.0, atol=1e-6)

    def test_output_stays_in_unit_range(self):
        """合法输入下输出不应越界到 [0, 1] 之外。"""
        rng = np.random.default_rng(seed=42)
        alpha = rng.random((32, 32), dtype=np.float32)
        depth = rng.random((32, 32), dtype=np.float32)

        weight = compute_sharpness_weight(alpha, depth)

        assert weight.dtype == np.float32
        assert weight.min() >= 0.0
        assert weight.max() <= 1.0

    def test_shape_mismatch_raises(self):
        """形状不一致必须立刻报错，不能让 numpy 静默广播。"""
        alpha = np.zeros((4, 4), dtype=np.float32)
        depth = np.zeros((5, 5), dtype=np.float32)

        with pytest.raises(ValueError, match="形状"):
            compute_sharpness_weight(alpha, depth)


class TestComposeDepthOfField:
    """验证按权重混合原图与模糊图的行为。"""

    def test_weight_one_keeps_original(self):
        """权重全 1 时结果应与原图完全一致（一点都不能糊）。"""
        image = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        weight = np.ones((16, 16), dtype=np.float32)

        result = compose_depth_of_field(image, weight, blur_sigma=10.0)

        np.testing.assert_array_equal(result, image)

    def test_flat_image_unchanged_by_any_weight(self):
        """纯色图模糊后还是纯色，所以无论权重如何结果都不变。"""
        image = np.full((16, 16, 3), 128, dtype=np.uint8)
        weight = np.zeros((16, 16), dtype=np.float32)

        result = compose_depth_of_field(image, weight, blur_sigma=10.0)

        np.testing.assert_array_equal(result, image)

    def test_weight_zero_blurs(self):
        """权重全 0 时应完全取模糊图，中心像素被邻域拉平。"""
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[16, 16] = 255  # 在中心放一个亮点
        weight = np.zeros((32, 32), dtype=np.float32)

        result = compose_depth_of_field(image, weight, blur_sigma=5.0)

        # 亮点被扩散：中心不再是 255，但周围不再是 0
        assert result[16, 16, 0] < 255
        assert result[16, 17, 0] > 0

    def test_output_dtype_and_range(self):
        """返回值必须是合法的 uint8 图像，不能因溢出回绕。"""
        image = np.full((8, 8, 3), 250, dtype=np.uint8)
        weight = np.ones((8, 8), dtype=np.float32)

        result = compose_depth_of_field(image, weight, blur_sigma=3.0)

        assert result.dtype == np.uint8
        assert result.shape == (8, 8, 3)
        # 若缺少 clip，250 附近的浮点误差可能回绕成很小的值
        assert result.min() >= 240

    def test_rejects_non_bgr_input(self):
        """灰度图应被拒绝，否则广播会得到错误结果而不报错。"""
        image = np.zeros((8, 8), dtype=np.uint8)
        weight = np.ones((8, 8), dtype=np.float32)

        with pytest.raises(ValueError, match="BGR"):
            compose_depth_of_field(image, weight, blur_sigma=3.0)
