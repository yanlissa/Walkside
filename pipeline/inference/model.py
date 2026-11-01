from __future__ import annotations

from importlib import import_module
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from pipeline.inference.architecture import (
    CLASS_NAMES,
    DEFAULT_MODEL_CONFIG,
    TileSegmentationModel,
)


LogCallback = Callable[[str], None]


def _safe_call(callback, default=None):
    try:
        return callback()
    except Exception:
        return default


def torch_diagnostics() -> dict:
    cuda_available = bool(_safe_call(torch.cuda.is_available, False))
    cuda_device_count = int(_safe_call(torch.cuda.device_count, 0) or 0)

    cuda_is_built = _safe_call(
        lambda: bool(torch.backends.cuda.is_built()),
        torch.version.cuda is not None,
    )

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_is_built": bool(cuda_is_built),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "torch_cuda_arch_list": _safe_call(torch.cuda.get_arch_list, []) or [],
    }


def format_torch_diagnostics(diagnostics: dict) -> str:
    runtime = diagnostics.get("torch_cuda_runtime") or "нет (CPU-сборка Torch)"
    frozen_text = "EXE" if diagnostics.get("frozen") else "Python"

    return (
        f"Режим: {frozen_text}; Python {diagnostics.get('python')}; "
        f"Torch {diagnostics.get('torch')}; CUDA runtime: {runtime}; "
        f"CUDA в сборке: {diagnostics.get('cuda_is_built')}; "
        f"CUDA доступна: {diagnostics.get('cuda_available')}; "
    )


def resolve_device(device_preference: str = "cuda") -> tuple[torch.device, dict]:
    preference = (device_preference or "cuda").strip().lower()
    diagnostics = torch_diagnostics()

    if preference == "cpu":
        return torch.device("cpu"), diagnostics

    if preference == "auto":
        device = torch.device("cuda" if diagnostics["cuda_available"] else "cpu")
        return device, diagnostics

    if preference != "cuda":
        raise ValueError(f"Неизвестный режим вычислений: {device_preference}")

    if diagnostics["cuda_available"]:
        return torch.device("cuda:0"), diagnostics

    if diagnostics["torch_cuda_runtime"] is None:
        reason = (
            "Установлена CPU-сборка PyTorch. В окружении сборки EXE должен быть "
            "torch из CUDA-репозитория PyTorch, а не пакет с PyPI для CPU."
        )
    elif diagnostics["cuda_device_count"] == 0:
        reason = (
            "CUDA-сборка PyTorch установлена, но процесс не видит NVIDIA GPU. "
            "Проверьте драйвер NVIDIA и наличие CUDA DLL внутри сборки."
        )
    else:
        reason = (
            "CUDA-сборка PyTorch установлена, однако инициализация CUDA завершилась "
            "ошибкой. Проверьте драйвер и совместимость GPU с этой сборкой Torch."
        )

    raise RuntimeError(
        "Выбран обязательный режим GPU, но torch.cuda.is_available() == False. "
        f"{reason}\nДиагностика: {format_torch_diagnostics(diagnostics)}"
    )


def verify_cuda_runtime(device: torch.device) -> None:
    """Принудительно загружает CUDA DLL и запускает короткую GPU-операцию."""
    if device.type != "cuda":
        return

    try:
        torch.cuda.init()
        left = torch.ones((64, 64), device=device)
        right = torch.ones((64, 64), device=device)
        result = left @ right
        torch.cuda.synchronize(device)

        if float(result[0, 0].item()) != 64.0:
            raise RuntimeError("CUDA smoke test вернул неверный результат")
    except Exception as exc:
        diagnostics = torch_diagnostics()
        raise RuntimeError(
            "CUDA обнаружена, но тестовая операция на GPU не выполнилась. "
            "В EXE обычно это означает, что не были упакованы torch/CUDA DLL, "
            "либо установлен несовместимый NVIDIA-драйвер.\n"
            f"Диагностика: {format_torch_diagnostics(diagnostics)}\n"
            f"Исходная ошибка: {exc}"
        ) from exc


def read_checkpoint(weights_path) -> Mapping:
    path = Path(weights_path)

    try:
        scalar_type = import_module("numpy._core.multiarray").scalar
    except (ImportError, AttributeError):
        scalar_type = import_module("numpy.core.multiarray").scalar
        
    numpy_safe_types = [
        scalar_type,
        (scalar_type, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float64")),
        type(np.dtype("float32")),
    ]
    with torch.serialization.safe_globals(numpy_safe_types):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint должен содержать словарь с параметрами модели")
    if "class_names" in checkpoint and tuple(checkpoint["class_names"]) != CLASS_NAMES:
        raise ValueError(
            f"В checkpoint другой порядок классов: {checkpoint['class_names']}"
        )
    return checkpoint


def build_model(
    num_classes: int = 4,
    device: torch.device | None = None,
    config: Mapping | None = None,
):
    if num_classes != len(CLASS_NAMES):
        raise ValueError(f"TileSegmentationModel рассчитана на {len(CLASS_NAMES)} класса")
    if device is None:
        device, _ = resolve_device("auto")

    options = dict(DEFAULT_MODEL_CONFIG if config is None else config)
    return TileSegmentationModel(**options).to(device)


def load_weights_to_library_model(
    model,
    weights_path,
    device: torch.device,
    log_callback: LogCallback | None = None,
    checkpoint: Mapping | None = None,
):
    log = log_callback or print
    log(f"Загрузка весов: {weights_path}")
    if checkpoint is None:
        checkpoint = read_checkpoint(weights_path)

    state_dict = checkpoint
    for field in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint.get(field), Mapping):
            state_dict = checkpoint[field]
            break

    cleaned_state = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise TypeError(f"Некорректный параметр в state_dict: {key}")

        clean_key = key
        while clean_key.startswith(("module.", "model.", "_orig_mod.")):
            clean_key = clean_key.split(".", 1)[1]
        if clean_key in cleaned_state:
            raise RuntimeError(f"Повторяющийся параметр после очистки: {clean_key}")
        cleaned_state[clean_key] = value

    expected = model.state_dict()
    missing = [
        key for key in expected
        if key not in cleaned_state and not key.endswith("num_batches_tracked")
    ]
    unexpected = [key for key in cleaned_state if key not in expected]
    wrong_shapes = [
        key for key in cleaned_state
        if key in expected and cleaned_state[key].shape != expected[key].shape
    ]
    if missing or unexpected or wrong_shapes:
        raise RuntimeError(
            "Веса несовместимы с TileSegmentationModel: "
            f"нет ключей {missing[:10]}, "
            f"лишние ключи {unexpected[:10]}, "
            f"размеры не совпадают {wrong_shapes[:10]}"
        )

    model.load_state_dict(cleaned_state, strict=False)
    log(f"Загружено параметров модели: {len(cleaned_state)}")
    return model


def load_inference_model(
    weights_path,
    num_classes: int = 4,
    device_preference: str = "auto",
    log_callback: LogCallback | None = None,
):
    log = log_callback or print
    cpu_device = torch.device("cpu")

    preference = (device_preference or "auto").strip().lower()
    candidate_device, diagnostics = resolve_device(preference)
    selected_device = cpu_device

    # Сначала проверяем CUDA отдельно, ещё до переноса модели.
    if candidate_device.type == "cuda":
        try:
            verify_cuda_runtime(candidate_device)
            selected_device = candidate_device
        except Exception:
            log(
                "CUDA доступна, но использовать аппаратное ускорение "
                "не удалось. Выполнение продолжено на CPU."
            )
            torch.backends.cudnn.benchmark = False
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    elif preference == "cpu":
        log("Используется CPU.")

    elif (
        diagnostics.get("cuda_is_built")
        or diagnostics.get("torch_cuda_runtime")
    ):
        log("CUDA недоступна. Выполнение продолжено на CPU.")

    else:
        log("CUDA недоступна. Используется CPU.")

    # Модель и параметры сначала загружаются на CPU.
    # Так ошибка CUDA не мешает прочитать файл параметров.
    checkpoint = read_checkpoint(weights_path)
    model_config = checkpoint.get("config", DEFAULT_MODEL_CONFIG)
    if not isinstance(model_config, Mapping):
        raise TypeError("Поле config в checkpoint должно быть словарём")
    log(f"Конфигурация новой модели: {dict(model_config)}")
    model = build_model(
        num_classes=num_classes,
        device=cpu_device,
        config=model_config,
    )

    model = load_weights_to_library_model(
        model=model,
        weights_path=weights_path,
        device=cpu_device,
        log_callback=log,
        checkpoint=checkpoint,
    )

    # После загрузки пытаемся перенести готовую модель на GPU.
    if selected_device.type == "cuda":
        try:
            model = model.to(selected_device)
            torch.cuda.synchronize(selected_device)
            torch.backends.cudnn.benchmark = True

            log(f"Аппаратное ускорение CUDA включено: ")

        except Exception:
            log(
                "CUDA доступна, но использовать аппаратное ускорение "
                "не удалось. Выполнение продолжено на CPU."
            )

            # После частично неудачного переноса безопаснее
            # создать чистую CPU-модель заново.
            del model

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            selected_device = cpu_device
            torch.backends.cudnn.benchmark = False

            model = build_model(
                num_classes=num_classes,
                device=cpu_device,
                config=model_config,
            )

            # При повторной загрузке не дублируем сообщения.
            model = load_weights_to_library_model(
                model=model,
                weights_path=weights_path,
                device=cpu_device,
                log_callback=lambda _message: None,
                checkpoint=checkpoint,
            )

    model.eval()
    return model, selected_device
