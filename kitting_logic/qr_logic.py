"""
QR code reading logic — no ROS, no hardware dependency.

Testable tonight with a plain webcam and your two printed (or on-screen) QR
codes. This is exactly the same code that will run tomorrow against the
lab's vision-kit camera; a camera is just a numbered device to OpenCV
either way.
"""
from typing import Optional

import cv2
import numpy as np

try:
    from pyzbar.pyzbar import decode as _zbar_decode
    HAVE_PYZBAR = True
except (ImportError, FileNotFoundError, OSError):
    # pyzbar needs the system 'libzbar0' library as well as the pip package.
    # If it's missing, fall back to OpenCV's built-in QR detector so you can
    # still test tonight — but install pyzbar for tomorrow, it's more robust.
    HAVE_PYZBAR = False
    _cv_qr_detector = cv2.QRCodeDetector()


def decode_qr(frame: np.ndarray) -> Optional[str]:
    """Return the text payload of the first QR code found in `frame`, or None."""
    if frame is None:
        return None

    if HAVE_PYZBAR:
        results = _zbar_decode(frame)
        if results:
            return results[0].data.decode("utf-8").strip()
        return None

    # OpenCV fallback path
    data, points, _ = _cv_qr_detector.detectAndDecode(frame)
    if data:
        return data.strip()
    return None


def normalize_mode(raw_text: Optional[str]) -> Optional[str]:
    """
    Turn whatever text was encoded in the QR code into a canonical mode key
    like 'mode_1'. Accepts a few likely spellings so a slightly different
    QR generator/format doesn't break detection during the demo.
    """
    if not raw_text:
        return None
    text = raw_text.strip().lower().replace("-", "_").replace(" ", "_")
    if text in ("mode_1", "mode1", "1"):
        return "mode_1"
    if text in ("mode_2", "mode2", "2"):
        return "mode_2"
    return text  # unrecognized — caller decides what to do with it
