# OmniRAG

import os
import sys


def _bootstrap_windows_dlls():
    """Windows 下 pywin32 的 DLL 目录默认不在 sys.path 中。

    langchain / mcp 的 client 模块在 Windows 上会 import pywintypes，
    因此在任何 app 子模块导入前先把 pywin32_system32 加入 DLL 搜索路径，
    兼容 vendored .libs 与常规 pip 安装两种场景。
    """
    if sys.platform != "win32":
        return
    for entry in list(sys.path):
        dll_dir = os.path.join(entry, "pywin32_system32")
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except (OSError, ValueError):
                pass
            # vendored 布局下还需要这两个路径才能 import pywintypes：
            #  - win32\_win32sysloader.pyd
            #  - win32\lib\pywintypes.py（DLL 加载桩）
            for sub in ("win32", os.path.join("win32", "lib")):
                sub_path = os.path.join(entry, sub)
                if os.path.isdir(sub_path) and sub_path not in sys.path:
                    sys.path.insert(0, sub_path)
            break


_bootstrap_windows_dlls()
