"""Настройка окружения до импорта Qt и PyTorch.

В frozen-сборках PyInstaller CUDA DLL лежат внутри каталога приложения, но
Windows не всегда добавляет эти каталоги в путь поиска DLL автоматически.
Этот модуль должен импортироваться из ``main.py`` раньше любого импорта torch.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable


_DLL_DIRECTORY_HANDLES: list[object] = []


def _unique_existing_directories(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue

        key = os.path.normcase(str(resolved))
        if key in seen or not resolved.is_dir():
            continue

        seen.add(key)
        result.append(resolved)

    return result


def _application_roots() -> list[Path]:
    roots: list[Path] = []

    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        roots.append(Path(bundle_root))

    roots.append(Path(__file__).resolve().parent)
    return _unique_existing_directories(roots)


def _source_torch_root() -> Path | None:
    """Находит torch без его импорта, чтобы DLL-пути настроить заранее."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, AttributeError, ValueError):
        return None

    if spec is None or spec.origin is None:
        return None

    return Path(spec.origin).resolve().parent


def native_library_directories() -> list[Path]:
    candidates: list[Path] = []

    for root in _application_roots():
        candidates.extend(
            [
                root,
                root / "_internal",

                # Torch/CUDA
                root / "torch" / "lib",
                root / "_internal" / "torch" / "lib",

                # Pyogrio/GDAL
                root / "pyogrio.libs",
                root / "_internal" / "pyogrio.libs",
            ]
        )

        # Новые CUDA wheels могут содержать отдельные пакеты nvidia.*.
        for nvidia_root in (root / "nvidia", root / "_internal" / "nvidia"):
            if nvidia_root.is_dir():
                candidates.append(nvidia_root)
                candidates.extend(nvidia_root.glob("*/bin"))
                candidates.extend(nvidia_root.glob("*/lib"))

    torch_root = _source_torch_root()
    if torch_root is not None:
        candidates.extend([torch_root, torch_root / "lib"])

        site_packages = torch_root.parent
        nvidia_root = site_packages / "nvidia"
        if nvidia_root.is_dir():
            candidates.append(nvidia_root)
            candidates.extend(nvidia_root.glob("*/bin"))
            candidates.extend(nvidia_root.glob("*/lib"))

    return _unique_existing_directories(candidates)


def configure_native_library_search() -> list[str]:
    """Добавляет каталоги PyTorch/CUDA в поиск DLL Windows.

    Возвращает фактически добавленные пути. На других ОС функция безопасна.
    Дескрипторы ``os.add_dll_directory`` сохраняются глобально: при их закрытии
    Windows немедленно исключает каталог из поиска DLL.
    """
    if sys.platform != "win32":
        return []

    directories = native_library_directories()
    added: list[str] = []

    current_path = os.environ.get("PATH", "")
    current_parts = {
        os.path.normcase(part)
        for part in current_path.split(os.pathsep)
        if part
    }

    prepend_to_path: list[str] = []

    for directory in directories:
        path_text = str(directory)
        normalized = os.path.normcase(path_text)

        if normalized not in current_parts:
            prepend_to_path.append(path_text)
            current_parts.add(normalized)

        if hasattr(os, "add_dll_directory"):
            try:
                handle = os.add_dll_directory(path_text)
            except OSError:
                continue
            _DLL_DIRECTORY_HANDLES.append(handle)

        added.append(path_text)

    if prepend_to_path:
        os.environ["PATH"] = os.pathsep.join(prepend_to_path + [current_path])

    return added


def set_windows_app_user_model_id(app_id: str) -> bool:
    """Задаёт стабильный AppUserModelID для иконки и прогресса taskbar."""
    if sys.platform != "win32":
        return False

    try:
        import ctypes

        setter = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        setter.argtypes = [ctypes.c_wchar_p]
        setter.restype = ctypes.c_long
        result = setter(str(app_id))
        return int(result) == 0
    except Exception:
        return False
