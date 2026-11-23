"""``paths.describe_control_chars`` 的单元测试。

这些测试来自一个真实事故：Windows 路径 ``study\\2.DepthAnything`` 里的 ``\\2``
是合法的八进制转义，Python 会把它变成码位 2 的控制字符。路径悄悄少一个字符，
``is_dir()`` 返回 False，而报错里的路径看着和写的一模一样 —— 因为控制字符不显示。
"""

from __future__ import annotations

from pathlib import Path

from dof_composer.paths import _repair_path, describe_control_chars


class TestDescribeControlChars:
    def test_clean_path_returns_none(self):
        """正常路径不该被误报。"""
        assert describe_control_chars(r"F:\VsCodeProjects\Pytorch\study") is None

    def test_forward_slash_path_returns_none(self):
        """正斜杠写法本身就不会产生控制字符。"""
        assert describe_control_chars("F:/VsCodeProjects/Pytorch/study") is None

    def test_octal_escape_is_detected(self):
        """``\\2`` 这类八进制转义必须被抓到。"""
        # 用 chr(2) 显式构造"已被转义污染"的字符串，避免测试文件自己触发 SyntaxWarning
        broken = "F:\\VsCodeProjects\\Pytorch\\study\\" + chr(2) + ".DepthAnything"
        message = describe_control_chars(broken)

        assert message is not None
        assert "\\x02" in message
        assert "控制字符" in message

    def test_report_explains_the_cause(self):
        """诊断要说清是反斜杠被当成了转义符，而不只是"路径不存在"。"""
        message = describe_control_chars("F:\\VsCodeProjects\\Pytorch\\study\\" + chr(2) + ".x")

        assert message is not None
        assert "转义" in message
        assert "八进制" in message

    def test_multiple_control_chars_are_counted(self):
        """多处出错时要报总数，否则用户改完第一处还会撞第二处。"""
        broken = "F:\\" + chr(1) + ".a\\" + chr(2) + ".b"
        message = describe_control_chars(broken)

        assert message is not None
        assert "另有" in message


class TestRepairPath:
    def test_repairs_octal_escape_to_existing_path(self, tmp_path: Path):
        """能反推出真实存在路径时，应给出可直接复制的写法。"""
        target = tmp_path / "2.DepthAnything"
        target.mkdir()

        # 构造一条"被转义污染"的路径：把 \2 换成真正的 chr(2)
        broken = str(tmp_path) + "\\" + chr(2) + ".DepthAnything"

        repaired = _repair_path(broken)

        assert repaired is not None
        assert ".DepthAnything" in repaired
        assert Path(repaired).is_dir()

    def test_returns_none_when_no_variant_exists(self):
        """猜不出真实路径时不能瞎编一个给用户。"""
        assert _repair_path("F:\x02.not_a_real_dir_xyz") is None
