# utils/runtime_env.py
"""
运行环境自愈（conda 激活等价物）
================================

**背景（实测根因）**：`启动药尘光.bat` 为了让用户不必把 conda 放进 PATH，
会直接调用环境里的解释器：

    "%PY%" -m streamlit run app.py      # PY = <env>\\python.exe

这**跳过了 `conda activate`**，于是缺了两件对本项目很关键的事：

1. `<env>\\Library\\bin`（Windows 下 `smina.exe`、`openbabel-3.dll` 就在这里）、
   `<env>\\Scripts` 等目录**不在 PATH 上** → `shutil.which("smina")` 返回 None，
   对接页误报"未检测到 Smina 可执行文件"；
2. conda 的 `etc/conda/activate.d/openbabel_activate-env_vars.bat` **没执行** →
   `BABEL_DATADIR` 未设置 → Open Babel 找不到自己的数据/插件目录，
   连 `pdb` / `sdf` / `mol2` 这些基础格式都注册不上，
   `pybel.readfile("pdb", ...)` 会直接抛
   `ValueError: pdb is not a recognised Open Babel format`。

本模块把这两件事在 **Python 进程内**补齐，使应用不再依赖"必须用 conda activate 启动"。
函数是幂等的：已经设置过的东西不会被覆盖。

作者：dadamingli
"""

import logging
import os
import sys
from typing import Dict, List

logger = logging.getLogger(__name__)

#: 经过一次调用后就不再重复扫描
_APPLIED = False


def _candidate_path_dirs(prefix: str) -> List[str]:
    """返回 conda 激活时会加进 PATH 的目录（按 conda 的顺序）。"""
    return [
        os.path.join(prefix, "Library", "bin"),            # smina.exe / openbabel-3.dll
        os.path.join(prefix, "Scripts"),                   # pip console scripts
        prefix,                                            # python.exe 所在
        os.path.join(prefix, "Library", "mingw-w64", "bin"),
        os.path.join(prefix, "Library", "usr", "bin"),
        os.path.join(prefix, "bin"),                       # Linux/macOS 布局
    ]


def _apply_path(prefix: str) -> List[str]:
    """把环境目录补进 PATH 前面，返回实际新增的目录列表。"""
    existing = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    existing_lower = {p.rstrip("\\/").lower() for p in existing}

    added: List[str] = []
    for directory in _candidate_path_dirs(prefix):
        if not os.path.isdir(directory):
            continue
        if directory.rstrip("\\/").lower() in existing_lower:
            continue
        added.append(directory)

    if added:
        os.environ["PATH"] = os.pathsep.join(added + existing)
    return added


def _apply_babel_env(prefix: str) -> List[str]:
    """补齐 Open Babel 需要的数据/插件目录环境变量，返回新增的键值说明。"""
    applied: List[str] = []

    if not os.environ.get("BABEL_DATADIR"):
        # conda 的 activate.d 脚本用的是 <prefix>/share/openbabel；
        # 部分构建放在 Library/share/openbabel，两处都探一下。
        for candidate in (
            os.path.join(prefix, "share", "openbabel"),
            os.path.join(prefix, "Library", "share", "openbabel"),
        ):
            if os.path.isdir(candidate):
                os.environ["BABEL_DATADIR"] = candidate
                applied.append(f"BABEL_DATADIR={candidate}")
                break

    if not os.environ.get("BABEL_LIBDIR"):
        plugin_root = os.path.join(prefix, "Library", "lib", "openbabel")
        if os.path.isdir(plugin_root):
            # 目录下通常还有版本号子目录；Open Babel 接受父目录
            os.environ["BABEL_LIBDIR"] = plugin_root
            applied.append(f"BABEL_LIBDIR={plugin_root}")

    return applied


def ensure_conda_runtime_env(force: bool = False) -> Dict[str, object]:
    """补齐 conda 激活才会设置的运行环境（PATH / BABEL_*），幂等。

    Args:
        force: 忽略"只做一次"的缓存，强制重新检查。

    Returns:
        ``{"prefix": str, "path_added": [...], "env_set": [...]}``，
        便于调用方记录日志或在页面上排查。
    """
    global _APPLIED
    result: Dict[str, object] = {"prefix": None, "path_added": [], "env_set": []}
    if _APPLIED and not force:
        return result

    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    if not prefix or not os.path.isdir(prefix):
        _APPLIED = True
        return result

    result["prefix"] = prefix
    result["path_added"] = _apply_path(prefix)
    result["env_set"] = _apply_babel_env(prefix)

    if result["path_added"] or result["env_set"]:
        logger.info(
            "运行环境自愈：PATH 补充 %s；环境变量 %s",
            result["path_added"],
            result["env_set"],
        )
    _APPLIED = True
    return result
