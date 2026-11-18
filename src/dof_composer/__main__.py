"""允许 ``python -m dof_composer`` 直接运行。"""

from dof_composer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
