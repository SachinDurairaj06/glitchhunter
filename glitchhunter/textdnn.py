"""Text detection + recognition with OpenCV 5's DNN module (OpenCV Zoo PP-OCRv3 DB + CRNN).

Both networks load through cv.dnn.readNetFromONNX with an explicit engine, so the
new OpenCV 5 graph engine runs them by default (ENGINE_AUTO falls back to the
classic engine if a layer is unsupported).
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from .config import MODELS_DIR

MODEL_FILES = {
    "text_detection_en_ppocrv3_2023may.onnx": (
        "https://huggingface.co/opencv/text_detection_ppocr/resolve/main/text_detection_en_ppocrv3_2023may.onnx",
        "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587",
    ),
    "text_recognition_CRNN_EN_2021sep.onnx": (
        "https://huggingface.co/opencv/text_recognition_crnn/resolve/main/text_recognition_CRNN_EN_2021sep.onnx",
        "a84b1f6e11a65c2d733cb0cc1f014aae3f99051e3f11447dc282faa678eee544",
    ),
}
CHARSET_EN_36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_REC_TARGET = np.array([[0, 31], [0, 0], [99, 0], [99, 31]], np.float32)

ENGINES = {"auto": "ENGINE_AUTO", "new": "ENGINE_NEW", "classic": "ENGINE_CLASSIC"}


def engine_from_env() -> int:
    name = ENGINES.get(os.environ.get("GH_DNN_ENGINE", "auto").lower(), "ENGINE_AUTO")
    return getattr(cv.dnn, name)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_models(models_dir: Path = MODELS_DIR) -> list[Path]:
    """Download the OpenCV Zoo text models (pinned by SHA-256) if they are missing."""
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (url, digest) in MODEL_FILES.items():
        path = models_dir / name
        if not path.exists() or _sha256(path) != digest:
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            if _sha256(tmp) != digest:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch for {name}")
            tmp.replace(path)
        paths.append(path)
    return paths


def models_available(models_dir: Path = MODELS_DIR) -> bool:
    return all((models_dir / name).exists() for name in MODEL_FILES)


@dataclass
class TextBox:
    quad: np.ndarray  # 4x2 float32: bottom-left, top-left, top-right, bottom-right
    confidence: float
    text: str

    @property
    def rect(self) -> tuple[int, int, int, int]:
        x, y, w, h = cv.boundingRect(self.quad.astype(np.float32))
        return int(x), int(y), int(w), int(h)


class TextReader:
    """PP-OCRv3 text detector + CRNN recognizer; thread-safe via a lock."""

    def __init__(self, models_dir: Path = MODELS_DIR, engine: int | None = None):
        engine = engine_from_env() if engine is None else engine
        det_net = cv.dnn.readNetFromONNX(str(models_dir / "text_detection_en_ppocrv3_2023may.onnx"), engine)
        self.detector = cv.dnn.TextDetectionModel_DB(det_net)
        self.detector.setBinaryThreshold(0.3)
        self.detector.setPolygonThreshold(0.5)
        self.detector.setUnclipRatio(2.0)
        self.detector.setMaxCandidates(200)
        self.recognizer = cv.dnn.readNetFromONNX(str(models_dir / "text_recognition_CRNN_EN_2021sep.onnx"), engine)
        self.engine = engine
        self._lock = threading.Lock()

    def read(self, image: np.ndarray) -> list[TextBox]:
        h, w = image.shape[:2]
        size = (max(32, math.ceil(w / 32) * 32), max(32, math.ceil(h / 32) * 32))
        with self._lock:
            self.detector.setInputParams(1.0 / 255.0, size, (123.675, 116.28, 103.53), True)
            self.detector.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))
            quads, confidences = self.detector.detect(image)
            boxes = []
            for quad, conf in zip(quads, list(confidences) or [1.0] * len(quads)):
                q = np.asarray(quad, np.float32).reshape(4, 2)
                boxes.append(TextBox(q, float(conf), self._recognize(image, q)))
        return boxes

    def _recognize(self, image: np.ndarray, quad: np.ndarray) -> str:
        warp = cv.getPerspectiveTransform(quad, _REC_TARGET)
        crop = cv.warpPerspective(image, warp, (100, 32))
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        blob = cv.dnn.blobFromImage(gray, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
        self.recognizer.setInput(blob)
        out = self.recognizer.forward()
        chars, prev = [], -1
        for step in range(out.shape[0]):
            c = int(np.argmax(out[step][0]))
            if c != 0 and c != prev:
                chars.append(CHARSET_EN_36[c - 1])
            prev = c
        return "".join(chars)


_reader: TextReader | None = None
_reader_lock = threading.Lock()


def get_reader() -> TextReader | None:
    """Shared reader, or None when the model files are not present."""
    global _reader
    with _reader_lock:
        if _reader is None and models_available():
            _reader = TextReader()
        return _reader
