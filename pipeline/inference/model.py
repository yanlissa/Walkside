from __future__ import annotations

import platform
import sys
from typing import Callable

import torch

from pipeline.inference.architecture import get_seg_model


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


def build_model(num_classes: int = 4, device: torch.device | None = None):
    if device is None:
        device, _ = resolve_device("auto")

    model = get_seg_model(num_classes=num_classes)
    return model.to(device)


def load_weights_to_library_model(
    model,
    weights_path,
    device: torch.device,
    log_callback: LogCallback | None = None,
):
    log = log_callback or print
    log(f"Загрузка весов: {weights_path}")

    # Загрузка на CPU уменьшает пиковое потребление памяти GPU. load_state_dict
    # сам скопирует подходящие тензоры в параметры модели на выбранном device.
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Файл весов не содержит словарь state_dict")

    model_dict = model.state_dict()
    new_state_dict = {}

    loaded_keys = []
    skipped_keys = []

    for key, value in state_dict.items():
        clean_key = key.replace("model.", "").replace("backbone.", "").replace("module.", "")

        matched_key = None

        if clean_key in model_dict:
            matched_key = clean_key
        else:
            for model_key in model_dict.keys():
                if model_key.endswith(clean_key):
                    matched_key = model_key
                    break

        if matched_key is None:
            skipped_keys.append((key, "no matching key"))
            continue

        if not hasattr(value, "shape"):
            skipped_keys.append((key, "checkpoint value is not a tensor"))
            continue

        if model_dict[matched_key].shape != value.shape:
            skipped_keys.append((
                key,
                f"shape mismatch: ckpt {tuple(value.shape)} vs model {tuple(model_dict[matched_key].shape)}",
            ))
            continue

        new_state_dict[matched_key] = value
        loaded_keys.append(matched_key)

    model.load_state_dict(new_state_dict, strict=False)

    if not loaded_keys:
        raise RuntimeError(
            "Файл параметров несовместим с текущей версией приложения"
        )
    
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
    model = build_model(
        num_classes=num_classes,
        device=cpu_device,
    )

    model = load_weights_to_library_model(
        model=model,
        weights_path=weights_path,
        device=cpu_device,
        log_callback=log,
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
            )

            # При повторной загрузке не дублируем сообщения.
            model = load_weights_to_library_model(
                model=model,
                weights_path=weights_path,
                device=cpu_device,
                log_callback=lambda _message: None,
            )

    model.eval()
    return model, selected_device
