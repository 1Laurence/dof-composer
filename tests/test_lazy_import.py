"""``dof_composer.__getattr__`` 延迟导出与缺依赖提示的测试。

两条契约：

1. ``import dof_composer`` 不能把 torch 拖进来 —— 只用合成公式的人不该被迫装 2GB。
2. 真要导入需要 torch 的类时，缺依赖的报错必须**能照做**，
   而不是一句裸的 ``No module named 'torch'``。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import dof_composer


class TestLazyImport:
    def test_import_does_not_load_torch(self):
        """顶层 import 不应把 torch 带进来。

        必须开子进程测：本测试进程里 torch 早被别的测试导入过了，
        在当前进程里查 ``sys.modules`` 永远是 True，测不出问题。
        """
        code = textwrap.dedent(
            """
            import sys
            import dof_composer

            assert "torch" not in sys.modules, "import dof_composer 把 torch 拖进来了"
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_version_available_without_heavy_deps(self):
        """读版本号是最轻的用法，不该触发任何重量级导入。"""
        assert dof_composer.__version__ == "0.1.0"

    def test_unknown_attribute_raises_attribute_error(self):
        """拼错名字要抛 AttributeError，而不是别的类型。

        这点重要：hasattr() / getattr(x, n, default) 只吞 AttributeError。
        """
        # 用变量传名字，避免 ruff 把常量 getattr 判成"可以直接点号访问"
        missing_name = "this_symbol_does_not_exist"
        with pytest.raises(AttributeError, match="no attribute"):
            getattr(dof_composer, missing_name)

    def test_formula_importable_without_torch(self):
        """纯 numpy 的两个函数必须能取到。"""
        weight_fn = dof_composer.compute_sharpness_weight
        compose_fn = dof_composer.compose_depth_of_field

        assert callable(weight_fn)
        assert callable(compose_fn)

    def test_all_matches_lazy_exports(self):
        """``__all__`` 与实际可导出符号必须对得上，否则 ``from x import *`` 会炸。"""
        expected = {name for name in dof_composer.__all__ if name != "__version__"}

        assert expected == set(dof_composer._LAZY_EXPORTS)

    def test_dir_lists_only_public_api(self):
        """``dir()`` 只该列公开符号，不该漏出内部子模块与 import 残留。

        没有 ``__dir__`` 时，导入过 ``SubjectSegmenter`` 之后
        ``dir(dof_composer)`` 会冒出 ``segmenter`` / ``depth_estimator`` ——
        用户会误以为那是公开 API。
        """
        _ = dof_composer.compute_sharpness_weight  # 触发一次子模块导入

        listed = dir(dof_composer)

        assert set(listed) == set(dof_composer.__all__)
        assert "segmenter" not in listed
        assert "depth_estimator" not in listed
        assert "annotations" not in listed
        assert "importlib" not in listed


class TestMissingDependencyHint:
    """缺依赖时要用 ImportError 说清楚"装哪个包、怎么装"。"""

    def test_torch_hint_names_the_install_command(self):
        """.name == 'torch' 时必须给出 pip 命令与 CPU/CUDA 的区分。"""
        hint = dof_composer._INSTALL_HINTS["torch"]

        assert "pip install torch" in hint
        assert "cu126" in hint
        assert "cpu" in hint

    def test_hint_table_covers_the_heavy_deps(self):
        """两个模型类依赖的包都要在提示表里，否则会漏掉裸报错。"""
        for package in ("torch", "transformers", "PIL", "cv2", "numpy"):
            assert package in dof_composer._INSTALL_HINTS, f"{package} 缺提示"

    def test_submodule_missing_is_not_mistaken_for_absent_package(self):
        """``torch._C`` 缺失说明 torch 装了但坏了，不能报成"没装 torch"。

        提示表的键与 ``ModuleNotFoundError.name`` 精确匹配，
        所以 ``torch._C`` 查不到，会原样抛出。
        """
        assert "torch._C" not in dof_composer._INSTALL_HINTS

    def test_raises_import_error_with_hint(self, monkeypatch):
        """模拟 import 时缺 torch，应得到带安装命令的 ImportError。"""
        import importlib

        real_import = importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == "dof_composer.segmenter":
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake_import)

        with pytest.raises(ImportError) as excinfo:
            dof_composer._import_symbol(
                "dof_composer.segmenter", "SubjectSegmenter", "SubjectSegmenter"
            )

        message = str(excinfo.value)
        assert "torch" in message
        assert "pip install torch" in message
        # 还要告诉用户"只想用公式的话不需要装"
        assert "compute_sharpness_weight" in message

    def test_unrelated_missing_module_propagates_unchanged(self, monkeypatch):
        """不在提示表里的缺失应原样抛出，不要伪装成依赖安装问题。"""
        import importlib

        def fake_import(name, *args, **kwargs):
            raise ModuleNotFoundError("No module named 'weird_pkg'", name="weird_pkg")

        monkeypatch.setattr(importlib, "import_module", fake_import)

        with pytest.raises(ModuleNotFoundError, match="weird_pkg"):
            dof_composer._import_symbol("some.module", "X", "X")

    def test_hasattr_propagates_import_error(self, monkeypatch):
        """``hasattr`` 遇到缺依赖时会抛 ImportError，而不是返回 False。

        这是**有意的**：``hasattr`` 只吞 AttributeError，而这里抛的是 ImportError。
        若改成返回 False，``hasattr(dof_composer, "SubjectSegmenter")`` 会变成
        False —— 但符号明明存在，只是依赖没装，那才是真正的误导。

        锁住这个行为，避免以后有人为了"让 hasattr 安静"而把 ImportError 改成
        AttributeError，那样缺依赖就会伪装成"没有这个功能"。
        """
        import importlib

        def fake_import(name, *args, **kwargs):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")

        monkeypatch.setattr(importlib, "import_module", fake_import)

        with pytest.raises(ImportError):
            hasattr(dof_composer, "SubjectSegmenter")
