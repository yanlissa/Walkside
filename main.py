from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

from runtime_environment import (
    configure_native_library_search,
    set_windows_app_user_model_id,
)


_ADDED_NATIVE_LIBRARY_DIRS = configure_native_library_search()

from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMessageBox

from config import APP_INSTANCE_KEY, APP_TITLE, APP_USER_MODEL_ID


class SingleInstanceLock:
    def __init__(self, key: str):
        self.key = key
        self.server: QLocalServer | None = None

    def acquire(self) -> bool:
        socket = QLocalSocket()
        socket.connectToServer(self.key)

        if socket.waitForConnected(150):
            socket.disconnectFromServer()
            return False

        QLocalServer.removeServer(self.key)
        self.server = QLocalServer()
        return self.server.listen(self.key)


def run_gpu_diagnostics() -> int:
    """Проверяет CUDA в том же Python/EXE и сохраняет текстовый отчёт."""
    from pipeline.inference.model import (
        format_torch_diagnostics,
        resolve_device,
        torch_diagnostics,
        verify_cuda_runtime,
    )

    lines = [format_torch_diagnostics(torch_diagnostics())]
    lines.append("DLL search paths:")
    lines.extend(f"  {path}" for path in _ADDED_NATIVE_LIBRARY_DIRS)

    exit_code = 0
    try:
        device, _ = resolve_device("cuda")
        verify_cuda_runtime(device)
        lines.append("CUDA smoke test: OK")
    except Exception as exc:
        exit_code = 2
        lines.append(f"CUDA smoke test: FAILED: {exc}")

    text = "\n".join(lines)
    output_dir = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path.cwd()
    )
    output_path = output_dir / "gpu_diagnostics.txt"
    output_path.write_text(text, encoding="utf-8")

    print(text)
    print(f"Отчёт: {output_path}")
    return exit_code


def main() -> int:
    multiprocessing.freeze_support()
    set_windows_app_user_model_id(APP_USER_MODEL_ID)

    if "--gpu-diagnostics" in sys.argv:
        return run_gpu_diagnostics()

    # Ленивый импорт гарантирует, что DLL-пути уже настроены.
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setApplicationDisplayName(APP_TITLE)
    app.setOrganizationName("WalkSide")

    single_instance_lock = SingleInstanceLock(APP_INSTANCE_KEY)

    if not single_instance_lock.acquire():
        QMessageBox.warning(
            None,
            "Приложение уже открыто",
            "WalkSide Pipeline уже запущен. "
            "Одновременно можно открыть только один экземпляр.",
        )
        return 0

    app.single_instance_lock = single_instance_lock

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
