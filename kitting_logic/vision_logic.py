"""
"Is this item on the table?" logic — no ROS, no hardware dependency.

Each item is identified by a class name (configured in
config/checklists.yaml under item_classes), and we call it "present" if
some detector — a locally trained YOLOv8 model, or a hosted Roboflow
model called over the network, or both combined via CompositeDetector —
reports that class with at least the configured confidence. This replaced
an earlier HSV colour-blob approach, which broke down under lighting
changes and couldn't tell two same-coloured items apart.

Two ways to get a working detector for an item, see the README's
"Training the YOLOv8 model" section for the full steps:

  A) Use an existing public model over Roboflow's hosted inference API
     (RoboflowAPIDetector) — no local training at all. Needs an API key
     in config/secrets.yaml's roboflow.api_key (free, from
     app.roboflow.com/settings/api) and a model_id in
     config/checklists.yaml's roboflow.model_id.
  B) Train your own local model (YoloDetector) from photos — your own
     captures (tools/capture_training_images.py) and/or a public dataset,
     merged in Roboflow and exported/trained locally. Point
     config/checklists.yaml's yolo.weights at the resulting best.pt.

build_detector() wires up whichever of these are configured (both is
fine — different items can come from different backends) and raises
DetectorNotConfigured with setup steps if neither is available, rather
than silently detecting nothing.
"""
from typing import Dict, List, Optional, Protocol
import os

import numpy as np
import yaml


class Detector(Protocol):
    """Anything with this method can stand in for a real YOLO model — this
    is what lets tests (and the no-camera demo) run without loading any
    model, downloading weights, or touching a camera."""

    def detect(self, frame: np.ndarray) -> Dict[str, float]:
        """Return, for every class detected in `frame`, its highest
        confidence score (0.0-1.0). Classes not seen are simply absent
        from the dict."""
        ...


class DetectorNotConfigured(RuntimeError):
    """Raised when neither a local model nor a hosted API is set up yet."""
    pass


# Kept as an alias: earlier code/docs referred to this as YoloNotConfigured
# specifically. DetectorNotConfigured covers the Roboflow-API case too.
YoloNotConfigured = DetectorNotConfigured


class YoloDetector:
    """
    Thin wrapper around a locally-loaded Ultralytics YOLOv8 model. Loads
    the weights once — that's the expensive part — then .detect() is just
    a forward pass per frame.
    """

    def __init__(self, weights_path: str, min_predict_confidence: float = 0.1):
        if not os.path.exists(weights_path):
            raise DetectorNotConfigured(
                f"No YOLO weights found at {weights_path!r}. Train a model on "
                f"your own items and point config/checklists.yaml's yolo.weights "
                f"at it — see the steps at the top of kitting_logic/vision_logic.py."
            )
        try:
            from ultralytics import YOLO  # imported lazily: importing this module
            # (e.g. for tests, or the no-camera demo) shouldn't require
            # ultralytics/torch unless a real detector is actually built.
        except ImportError as e:
            raise DetectorNotConfigured(
                f"Could not import ultralytics ({e}). Run with the Python that "
                f"has it installed (e.g. `source .venv/bin/activate` first), or "
                f"`pip install -r requirements.txt`."
            )
        self._model = YOLO(weights_path)
        # Ask the model for anything even remotely plausible; per-item
        # confidence thresholds (below) do the real filtering. This just
        # avoids YOLO's own default cutoff discarding a box before we get
        # a chance to apply an item's configured threshold.
        self._min_predict_confidence = min_predict_confidence

    def detect(self, frame: np.ndarray) -> Dict[str, float]:
        results = self._model.predict(
            frame, conf=self._min_predict_confidence, verbose=False
        )[0]
        best: Dict[str, float] = {}
        names = results.names
        for box in results.boxes:
            name = names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf > best.get(name, 0.0):
                best[name] = conf
        return best


_detector_cache: Dict[str, "YoloDetector"] = {}


def get_detector(weights_path: str) -> YoloDetector:
    """Loads (and caches) a YoloDetector for a given weights file, so
    perception_node / demo_integration only pay the model-load cost once."""
    key = os.path.abspath(weights_path)
    if key not in _detector_cache:
        _detector_cache[key] = YoloDetector(weights_path)
    return _detector_cache[key]


class RoboflowAPIDetector:
    """
    Calls a model hosted on Roboflow's inference API instead of running
    one locally — for an item already covered by an existing public
    Roboflow model (see config/checklists.yaml's roboflow.model_id),
    training your own copy would just be redundant.

    Uses Roboflow's serverless REST endpoint directly via stdlib urllib
    rather than their `inference_sdk` package: that package caps at Python
    <3.13 (no wheel for 3.14+ at time of writing), which would make this
    an unreliable dependency depending on whoever's running the project.
    See https://docs.roboflow.com/guides/run-model-serverless-api for the
    endpoint's docs — POST to serverless.roboflow.com/{model_id} with the
    API key as a Bearer token (NOT a query param — that hits the older
    detect.roboflow.com contract and 401s with "not authorized for
    serverless inference").

    Needs network access at detect() time (one HTTP call per frame) — this
    trades local setup/training for a per-call round trip, which is fine
    for the ~1 check/second this project does but not for a tight loop.
    """

    def __init__(self, api_key: str, model_id: str):
        if not api_key:
            raise DetectorNotConfigured(
                "No Roboflow API key configured. Get a free one at "
                "app.roboflow.com/settings/api and put it in config/secrets.yaml's "
                "roboflow.api_key (see config/secrets.example.yaml)."
            )
        self._api_key = api_key
        self._model_id = model_id

    def detect(self, frame: np.ndarray) -> Dict[str, float]:
        import base64
        import json
        import urllib.request

        import cv2  # already a core project dependency

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return {}
        image_b64 = base64.b64encode(buf.tobytes())

        url = f"https://serverless.roboflow.com/{self._model_id}"
        request = urllib.request.Request(
            url,
            data=image_b64,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.loads(response.read())

        best: Dict[str, float] = {}
        for pred in result.get("predictions", []):
            name = pred["class"]
            conf = float(pred["confidence"])
            if conf > best.get(name, 0.0):
                best[name] = conf
        return best


class CompositeDetector:
    """
    Merges detections from several sub-detectors into one dict, so
    different items can be recognized by different backends — e.g.
    red_block/blue_block via a hosted Roboflow model, rubiks_cube via a
    locally trained one. Runs every sub-detector on the same frame each
    call; if the same class name comes back from more than one, the
    higher confidence wins.

    A network-backed sub-detector (RoboflowAPIDetector) can fail per-call
    for reasons that have nothing to do with the others — a bad/expired
    key, a timeout, Roboflow being briefly down. Those failures are caught
    and logged rather than raised, so one flaky backend degrades to "that
    backend's items are missing this frame" instead of crashing detection
    entirely (which would also stop a perfectly good local model from
    reporting anything).
    """

    def __init__(self, detectors: List[Detector]):
        self._detectors = detectors
        self._warned = set()  # detector types we've already logged a failure for

    def detect(self, frame: np.ndarray) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for d in self._detectors:
            try:
                detections = d.detect(frame)
            except Exception as e:
                if type(d) not in self._warned:
                    print(f"[vision] {type(d).__name__} failed, continuing without it: {e}")
                    self._warned.add(type(d))
                continue
            for name, conf in detections.items():
                if conf > merged.get(name, 0.0):
                    merged[name] = conf
        return merged


def build_detector(config: dict, secrets_path: str, weights_path: str) -> Detector:
    """
    Builds the detector used by perception_node/demo_integration from
    config/checklists.yaml (`config`, already parsed) plus
    config/secrets.yaml (`secrets_path`, for the Roboflow API key) and the
    resolved local weights path. Combines whichever backends are actually
    configured:
        - RoboflowAPIDetector, if config has a `roboflow.model_id` AND
          secrets.yaml has a `roboflow.api_key`
        - YoloDetector, if a weights file exists at `weights_path`
    Raises DetectorNotConfigured if neither is available — silently
    detecting nothing would be worse than a clear error here.
    """
    detectors: List[Detector] = []

    roboflow_cfg = config.get("roboflow")
    if roboflow_cfg and roboflow_cfg.get("model_id"):
        api_key = None
        if os.path.exists(secrets_path):
            with open(secrets_path) as f:
                secrets = yaml.safe_load(f) or {}
            api_key = secrets.get("roboflow", {}).get("api_key")
        if api_key:
            detectors.append(RoboflowAPIDetector(api_key, roboflow_cfg["model_id"]))

    if os.path.exists(weights_path):
        detectors.append(get_detector(weights_path))

    if not detectors:
        raise DetectorNotConfigured(
            "No detector backend is configured: no Roboflow API key in "
            f"{secrets_path!r} (see roboflow.api_key in "
            f"config/secrets.example.yaml) and no local model at "
            f"{weights_path!r} (see kitting_logic/vision_logic.py's top "
            f"comment for how to train one)."
        )
    return detectors[0] if len(detectors) == 1 else CompositeDetector(detectors)


def item_present(detections: Dict[str, float], class_name: str, confidence: float) -> bool:
    """True if `class_name` was detected with at least `confidence` score."""
    return detections.get(class_name, 0.0) >= confidence


def find_missing_items(
    detector: Detector,
    frame: np.ndarray,
    required_items: List[str],
    item_classes: Dict[str, dict],
    default_confidence: float = 0.5,
) -> List[str]:
    """
    Runs the detector once on `frame`, then for each required item name
    looks up its YOLO class name (and optional per-item confidence) in
    item_classes and returns the list of required items NOT found.
    item_classes[name] looks like:
        {"class_name": "red_block", "confidence": 0.5}   # confidence optional
    """
    detections = detector.detect(frame)
    missing = []
    for name in required_items:
        cfg = item_classes.get(name)
        if cfg is None:
            # No class configured for this item yet — treat as missing so
            # it's obvious in testing rather than silently always-passing.
            missing.append(name)
            continue
        class_name = cfg.get("class_name", name)
        confidence = cfg.get("confidence", default_confidence)
        if not item_present(detections, class_name, confidence):
            missing.append(name)
    return missing
