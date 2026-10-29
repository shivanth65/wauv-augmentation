"""Image-only effects. Images are OpenCV BGR uint8; geometry never enters here."""

import albumentations as A
import cv2
import numpy as np


def enabled(settings, name, rng):
    effect = settings[name]
    return effect["enabled"] and rng.random() < effect["probability"]


def apply_effects(image, settings, rng):
    x = image.astype(np.float32) / 255
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx, yy = xx / w, yy / h
    for name in ("brightness", "contrast", "gamma"):
        if enabled(settings, name, rng):
            s = settings[name]
            value = rng.uniform(s["min"], s["max"])
            if name == "brightness":
                x *= value
            elif name == "contrast":
                x = (x - 0.5) * value + 0.5
            else:
                x = np.clip(x, 0, 1) ** value
    for name in ("uneven_lighting", "shadows"):
        if enabled(settings, name, rng):
            cx, cy = rng.uniform(0, 1, 2)
            radius = rng.uniform(0.15, 0.5)
            spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2))
            amount = rng.uniform(0, settings[name]["strength"])
            sign = -1 if name == "shadows" else rng.choice([-1, 1])
            x *= 1 + sign * amount * spot[..., None]
    if enabled(settings, "vignette", rng):
        amount = rng.uniform(0, settings["vignette"]["strength"])
        x *= (1 - amount * np.clip(2 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2), 0, 1))[..., None]
    if enabled(settings, "underwater_tint", rng):
        amount = rng.uniform(0, settings["underwater_tint"]["strength"])
        # Vary blue versus green, while attenuating red; BGR channel order.
        green = rng.uniform(0, 1)
        x *= np.array([1 + amount * (1 - green), 1 + amount * green, 1 - amount])
    if enabled(settings, "haze", rng):
        amount = rng.uniform(0, settings["haze"]["strength"])
        water = np.array([0.55, rng.uniform(0.45, 0.65), 0.25])
        x = x * (1 - amount) + water * amount
    for name in ("noise", "grain"):
        if enabled(settings, name, rng):
            shape = x.shape if name == "noise" else (h, w, 1)
            x += rng.normal(0, rng.uniform(0, settings[name]["strength"]), shape)
    image = (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
    if enabled(settings, "blur", rng):
        sigma = rng.uniform(0, settings["blur"]["max_sigma"])
        if sigma > 0.01:
            image = cv2.GaussianBlur(image, (0, 0), sigma)
    # Seed Albumentations independently for reproducible optional effects.
    transforms = []
    if enabled(settings, "motion_blur", rng):
        k = settings["motion_blur"]["kernel"]
        transforms.append(A.MotionBlur(blur_limit=(k, k), p=1))
    if enabled(settings, "jpeg", rng):
        s = settings["jpeg"]
        transforms.append(A.ImageCompression(quality_range=(s["min"], s["max"]), p=1))
    if transforms:
        image = A.Compose(transforms, seed=int(rng.integers(2**31)))(image=image)["image"]
    return image
