"""
Utility functions for animal detection and recognition using YOLO.
"""
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
from typing import Tuple, List, Dict, Optional
from config import (
    ANIMAL_CLASSES,
    ANIMAL_FEATURES,
    CONFIDENCE_THRESHOLD,  # kept for backward compat
    DET_CONF_THRESHOLD,
    CLS_CONF_THRESHOLD,
    MIN_BOX_AREA,
    MODEL_PATH,
)


class AnimalDetector:
    """Animal detection and recognition using a two-stage pipeline:
    1) YOLO object detection (boxes) to localize all animals in frame
    2) Custom classification (your trained best.pt) on each crop for fine labels

    If the classifier isn't available, we fall back to detector labels.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the YOLO animal detector.
        
        Args:
            model_path: Path to a custom trained YOLO model (if available)
        """
        # Paths
        custom_model_path = model_path or os.path.join(MODEL_PATH, "best.pt")
        alt_root_best = os.path.join(os.path.dirname(__file__), os.pardir, "best.pt")
        alt_root_best = os.path.abspath(alt_root_best)

        # 1) Load detection model (always needed for boxes)
        self._load_coco_model()  # sets self.det_model via YOLO('yolov8n.pt')

        # 2) Load classification model (your trained best.pt)
        self.cls_model = None
        self.use_custom_model = False
        for path in [custom_model_path, alt_root_best]:
            if os.path.exists(path):
                try:
                    print(f"Loading custom classification model: {path}")
                    self.cls_model = YOLO(path)
                    self.use_custom_model = True
                    print("Custom classification model loaded!")
                    break
                except Exception as e:
                    print(f"Failed to load classifier at {path}: {e}")
        if not self.use_custom_model:
            print("Warning: custom classifier not found. Using detector labels only.")
        
        # Store animal classes and features
        self.animal_classes = ANIMAL_CLASSES
        self.animal_features = ANIMAL_FEATURES
    
    def _load_coco_model(self):
        """Load the YOLOv8 detection model for boxes (COCO-pretrained)."""
        try:
            # Nano model is lightweight and fast
            self.det_model = YOLO("yolov8n.pt")

            # COCO dataset animal class indices (YOLOv8 uses the same order)
            self.coco_animal_classes = {
                13: "bird",
                14: "cat",
                15: "dog",
                16: "horse",
                17: "sheep",
                18: "cow",
                19: "elephant",
                20: "bear",
                21: "zebra",
                22: "giraffe",
            }

            # Mapping from generic COCO labels to the extended animal list used
            # in our classification dataset. This allows us to display richer
            # names even when falling back to the COCO model.
            self.animal_mapping = {
                "bird": [
                    "eagle",
                    "owl",
                    "flamingo",
                    "sparrow",
                    "duck",
                    "swan",
                    "woodpecker",
                    "pelican",
                    "sandpiper",
                    "hornbill",
                    "pelecaniformes",
                ],
                "cat": ["cat", "lion", "tiger", "leopard", "cheetah"],
                "dog": ["dog", "fox", "wolf", "coyote"],
                "horse": ["horse", "zebra", "donkey"],
                "sheep": ["sheep", "goat"],
                "cow": ["cow", "buffalo", "ox"],
                "elephant": ["elephant"],
                "bear": ["bear"],
                "zebra": ["zebra"],
                "giraffe": ["giraffe"],
            }
        except Exception as e:
            print(f"Warning: Could not load detection YOLO model. Error: {e}")
            self.det_model = None
    
    def detect_animal(self, image: np.ndarray) -> Tuple[str, float, str]:
        """Detect all animals and draw boxes. If classifier is available, label
        each crop with the fine-grained class from your dataset. Returns the
        top-confidence detection for backward compatibility."""
        if self.det_model is None:
            return "Unknown", 0.0, "Detection model not available."

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        def expand_box(x1, y1, x2, y2, w, h, ratio=0.1):
            """Expand box by a ratio while clamping into image bounds."""
            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
            half_w = bw * (1 + ratio) / 2.0
            half_h = bh * (1 + ratio) / 2.0
            nx1 = clamp(int(cx - half_w), 0, w - 1)
            ny1 = clamp(int(cy - half_h), 0, h - 1)
            nx2 = clamp(int(cx + half_w), 0, w - 1)
            ny2 = clamp(int(cy + half_h), 0, h - 1)
            return nx1, ny1, nx2, ny2

        def resize_square(img: np.ndarray, size: int = 224) -> np.ndarray:
            """Letterbox to a square canvas then resize to keep aspect ratio stable."""
            ih, iw = img.shape[:2]
            if ih == 0 or iw == 0:
                return img
            s = max(ih, iw)
            canvas = np.zeros((s, s, 3), dtype=img.dtype)
            y = (s - ih) // 2
            x = (s - iw) // 2
            canvas[y:y+ih, x:x+iw] = img
            return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)

        try:
            det_res = self.det_model(image, verbose=False)
            det = det_res[0] if isinstance(det_res, (list, tuple)) else det_res

            if getattr(det, "boxes", None) is None or len(det.boxes) == 0:
                return "Unknown", 0.0, "No animal detected with sufficient confidence."

            top_conf, top_name = 0.0, "Unknown"

            for box in det.boxes:
                det_conf = float(box.conf[0])
                if det_conf < DET_CONF_THRESHOLD:
                    continue

                # Detector class name (fallback)
                det_cls = int(box.cls[0])
                det_name = getattr(self.det_model, "names", {}).get(det_cls, f"cls_{det_cls}")

                # Box coords
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = image.shape[:2]
                x1, y1 = clamp(x1, 0, w - 1), clamp(y1, 0, h - 1)
                x2, y2 = clamp(x2, 0, w - 1), clamp(y2, 0, h - 1)

                # Filter tiny detections
                if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                    continue

                label_name = det_name
                label_conf = det_conf

                # If classifier exists, classify the crop for fine label
                if self.cls_model is not None:
                    # Expand a bit and preprocess crop for classification stability
                    ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, ratio=0.15)
                    crop = image[ey1:ey2, ex1:ex2]
                    if crop.size > 0:
                        crop = resize_square(crop, 224)
                        cls_res = self.cls_model(crop, verbose=False)
                        cls = cls_res[0] if isinstance(cls_res, (list, tuple)) else cls_res
                        if hasattr(cls, "probs") and cls.probs is not None:
                            probs = cls.probs.data.cpu().numpy()
                            idx = int(probs.argmax())
                            label_conf = float(probs.max())
                            if label_conf < CLS_CONF_THRESHOLD:
                                label_conf = det_conf
                                label_name = det_name
                            else:
                            # Prefer model's names; fallback to configured classes
                                label_name = (
                                    getattr(self.cls_model, "names", {}).get(idx)
                                    or (self.animal_classes[idx] if idx < len(self.animal_classes) else det_name)
                                )

                # Draw box and label for this detection
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"{label_name} {label_conf:.2f}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                if label_conf > top_conf:
                    top_conf, top_name = label_conf, label_name

            feature = self.animal_features.get(top_name, f"This is a {top_name}.")
            return top_name, top_conf, feature
        except Exception as e:
            print(f"Detection error: {e}")
            return "Unknown", 0.0, "Detection error occurred."

    # ------------------------------------------------------------------
    # Utility to overlay text (kept for backward-compatibility with CLI GUI)
    # ------------------------------------------------------------------
    def draw_detection(
        self,
        frame: np.ndarray,
        animal_name: str,
        confidence: float,
        position: Tuple[int, int],
    ) -> np.ndarray:
        """Draw simple label and confidence text on *frame* at *position*."""
        cv2.putText(
            frame,
            f"Animal: {animal_name}",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (position[0], position[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        return frame


def initialize_camera() -> cv2.VideoCapture:
    """
    Initialize the camera for video capture.
    
    Returns:
        VideoCapture object
    """
    # Try different camera indices
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Set resolution (adjust as needed)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
    
    # If no camera found, raise an exception
    raise RuntimeError("No camera found. Please connect a camera to your device.")


def release_camera(cap: cv2.VideoCapture) -> None:
    """
    Release the camera properly.
    
    Args:
        cap: VideoCapture object to release
    """
    cap.release()
    cv2.destroyAllWindows()