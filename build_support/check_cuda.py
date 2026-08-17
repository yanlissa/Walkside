"""Предварительная проверка окружения, из которого будет собираться EXE.

Запуск:
    python build_support/check_cuda.py

Скрипт завершается кодом 2, если установлена CPU-сборка Torch, CUDA не видна
или простая операция на GPU не выполняется. Собирать EXE из такого окружения
не следует: PyInstaller упакует именно установленные в нём библиотеки.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_environment import configure_native_library_search

configure_native_library_search()

from pipeline.inference.model import (
    format_torch_diagnostics,
    resolve_device,
    torch_diagnostics,
    verify_cuda_runtime,
)


def main() -> int:
    diagnostics = torch_diagnostics()
    print(format_torch_diagnostics(diagnostics))

    try:
        device, _ = resolve_device("cuda")
        verify_cuda_runtime(device)
    except Exception as exc:
        print(f"\nCUDA CHECK FAILED\n{exc}", file=sys.stderr)
        return 2

    print("\nCUDA CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
