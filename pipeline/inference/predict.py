from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


CLASS_TO_COLOR = {
    0: (61, 61, 245),
    1: (10, 153, 0),
    2: (255, 4, 4),
    3: (0, 0, 0),
}


def class_mask_to_color_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_idx, color in CLASS_TO_COLOR.items():
        color_mask[mask == class_idx] = color

    return color_mask


def preprocess_image(image_path):
    image_path = Path(image_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать картинку: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image_float = image.astype(np.float32) / 255.0
    image_norm = (image_float - mean) / std

    image_tensor = torch.from_numpy(np.ascontiguousarray(image_norm)).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0)

    return original_image, image_tensor


def predict_mask(model, device, image_path):
    original_image, image_tensor = preprocess_image(image_path)
    if device.type == "cuda":
        image_tensor = image_tensor.pin_memory().to(device, non_blocking=True)
    else:
        image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        output = model(image_tensor)

        if isinstance(output, (list, tuple)):
            output = output[0]

        output = F.interpolate(
            output,
            size=(original_image.shape[0], original_image.shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return original_image, pred_mask


def run_model_on_images(
    model,
    device,
    images_dir,
    labels_dir,
    skip_existing=True,
    pause_wait=None,
    stop_requested=None,
    progress_callback=None,
    image_paths=None,
    image_items=None,
):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    labels_dir.mkdir(parents=True, exist_ok=True)

    if image_items is None:
        if image_paths is None:
            image_paths = sorted([
                path for path in images_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ])
        else:
            image_paths = sorted({
                Path(path)
                for path in image_paths
                if Path(path).is_file()
                and Path(path).suffix.lower() in {".jpg", ".jpeg", ".png"}
            })

        image_items = [
            {
                "path": image_path,
                "label_stem": image_path.stem,
            }
            for image_path in image_paths
        ]
    else:
        image_items = sorted(
            [
                {
                    **item,
                    "path": Path(item["path"]),
                }
                for item in image_items
                if Path(item["path"]).is_file()
                and Path(item["path"]).suffix.lower()
                in {".jpg", ".jpeg", ".png"}
            ],
            key=lambda item: item["label_stem"],
        )

    saved = []
    skipped = []
    failed = []

    total_images = len(image_items)

    if progress_callback is not None:
        progress_callback(0, total_images)

    for image_index, image_item in enumerate(image_items, start=1):
        image_path = image_item["path"]
        if stop_requested is not None and stop_requested():
            return {
                "cancelled": True,
                "labels_dir": labels_dir,
                "saved": saved,
                "skipped": skipped,
                "failed": failed,
            }

        if pause_wait is not None and not pause_wait():
            return {
                "cancelled": True,
                "labels_dir": labels_dir,
                "saved": saved,
                "skipped": skipped,
                "failed": failed,
            }

        label_path = labels_dir / f"{image_item['label_stem']}.png"

        if skip_existing and label_path.exists():
            skipped.append(label_path)
            if progress_callback is not None:
                progress_callback(image_index, total_images)
            continue

        try:
            _, pred_mask = predict_mask(
                model=model,
                device=device,
                image_path=image_path,
            )

            color_mask = class_mask_to_color_mask(pred_mask)
            Image.fromarray(color_mask).save(label_path)

            saved.append(label_path)

        except Exception as exc:
            failed.append({
                "image_path": image_path,
                "error": str(exc),
            })

            print(f"FAILED mask: {image_path.name}: {exc}")

        if progress_callback is not None:
            progress_callback(image_index, total_images)

    return {
        "cancelled": False,
        "labels_dir": labels_dir,
        "saved": saved,
        "skipped": skipped,
        "failed": failed,
    }
