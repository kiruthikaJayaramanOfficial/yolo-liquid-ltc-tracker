"""src/fog_utils.py — fog simulation + schedule."""
import cv2
import numpy as np


def apply_fog(img: np.ndarray, intensity: float) -> np.ndarray:
    """Koschmieder's law: I(x) = I0·e^(-βd) + L·(1-e^(-βd))"""
    if intensity <= 0:
        return img.copy()
    h, w   = img.shape[:2]
    depth  = np.linspace(0.1, 1.0, h)[:, None] * np.ones((1, w))
    trans  = np.exp(-intensity * 3.5 * depth)[..., None]
    foggy  = img.astype(np.float32) * trans + 215 * (1 - trans)
    blur_k = max(1, int(intensity * 9) | 1)
    return cv2.GaussianBlur(foggy, (blur_k, blur_k), 0).clip(0, 255).astype(np.uint8)


def build_fog_schedule(n: int) -> np.ndarray:
    """Clear → ramp → peak 92% → clear schedule over n frames."""
    fog = np.zeros(n)
    a, b = int(0.25 * n), int(0.50 * n)
    c, d = int(0.70 * n), int(0.85 * n)
    fog[a:b] = np.linspace(0, 0.92, b - a)
    fog[b:c] = 0.92
    fog[c:d] = np.linspace(0.92, 0.0, d - c)
    return fog


def fog_category(fog: float) -> str:
    if fog < 0.15: return "Clear"
    if fog < 0.35: return "Light Haze"
    if fog < 0.55: return "Moderate Fog"
    if fog < 0.75: return "Dense Fog"
    return "Extreme Fog"
