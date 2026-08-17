"""PyInstaller runtime hook: настраивает поиск Torch/CUDA DLL до импорта torch."""

from runtime_environment import configure_native_library_search


# Сохраняем возвращаемое значение в глобальной переменной runtime hook, чтобы
# дескрипторы os.add_dll_directory оставались открытыми до завершения процесса.
WALKSIDE_NATIVE_LIBRARY_PATHS = configure_native_library_search()
