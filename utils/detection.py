"""
Utility functions for animal detection and recognition using YOLO.
"""
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
from typing import Tuple, List, Dict, Optional
from config import ANIMAL_CLASSES, ANIMAL_FEATURES, CONFIDENCE_THRESHOLD, MODEL_PATH


class AnimalDetector:
    """Animal detection and recognition system using YOLO."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the YOLO animal detector.
        
        Args:
            model_path: Path to a custom trained YOLO model (if available)
        """
        # Try to load custom trained model first
        custom_model_path = model_path or os.path.join(MODEL_PATH, "animal_detector.pt")
        
        if os.path.exists(custom_model_path):
            print(f"Loading custom trained model: {custom_model_path}")
            try:
                self.model = YOLO(custom_model_path)
                self.use_custom_model = True
                print("Custom model loaded successfully!")
            except Exception as e:
                print(f"Failed to load custom model: {e}")
                print("Falling back to COCO pre-trained model...")
                self._load_coco_model()
        else:
            print("Custom trained model not found, using COCO pre-trained model...")
            self._load_coco_model()
        
        # Store animal classes and features
        self.animal_classes = ANIMAL_CLASSES
        self.animal_features = ANIMAL_FEATURES
    
    def _load_coco_model(self):
        """Load the COCO pre-trained model as fallback."""
        try:
            self.model = YOLO('yolov8n.pt')  # Using nano version for speed
            self.use_custom_model = False
            
            # COCO dataset animal class indices
            # These are the indices in COCO dataset that correspond to animals
            self.coco_animal_classes = {
                13: 'bird', 14: 'cat', 15: 'dog', 16: 'horse', 17: 'sheep', 
                18: 'cow', 19: 'elephant', 20: 'bear', 21: 'zebra', 22: 'giraffe'
            }
            
            # Map COCO animals to our extended animal dataset
            self.animal_mapping = {
                'bird': ['eagle', 'owl', 'flamingo', 'sparrow', 'duck', 'swan', 'woodpecker', 
                        'pelican', 'sandpiper', 'hornbill', 'pelecaniformes'],
                'cat': ['cat', 'lion', 'tiger', 'leopard', 'cheetah'],
                'dog': ['dog', 'fox', 'wolf', 'coyote'],
                'horse': ['horse', 'zebra', 'donkey'],
                'sheep': ['sheep', 'goat'],
                'cow': ['cow', 'buffalo', 'ox'],
                'elephant': ['elephant'],
                'bear': ['bear'],
                'zebra': ['zebra'],
                'giraffe': ['giraffe']
            }
        except Exception as e:
            print(f"Warning: Could not load YOLO model. Error: {e}")
            self.model = None
            self.use_custom_model = False
    
    def detect_animal(self, image: np.ndarray) -> Tuple[str, float, str]:
        """
        Detect and identify an animal in the given image using YOLO.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple containing (animal_name, confidence, animal_features)
        """
        # If YOLO model is not available, return default
        if self.model is None:
            return "Unknown", 0.0, "Detection model not available."
        
        try:
            # Perform inference with YOLO
            results = self.model(image)
            
            # If using custom model, get classification results
            if self.use_custom_model:
                # For classification models, results are different
                if hasattr(results, 'probs') and results.probs is not None:
                    # Get probabilities
                    probs = results.probs.data.cpu().numpy()
                    class_idx = int(probs.argmax())
                    confidence = float(probs.max())
                    
                    # Get animal name from our dataset
                    if class_idx < len(self.animal_classes):
                        animal_name = self.animal_classes[class_idx]
                        animal_feature = self.animal_features.get(animal_name, f"This is a {animal_name}.")
                        return animal_name, confidence, animal_feature
                    else:
                        return "Unknown", confidence, "Animal class not recognized."
                else:
                    return "Unknown", 0.0, "No detection results."
            else:
                # Using COCO pre-trained model
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        # Get the highest confidence detection
                        for box in boxes:
                            # Get class id and confidence
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Check if it's an animal class we're interested in
                            if class_id in self.coco_animal_classes and confidence >= CONFIDENCE_THRESHOLD:
                                coco_animal = self.coco_animal_classes[class_id]
                                
                                # Map to our animal dataset
                                possible_animals = self.animal_mapping.get(coco_animal, [coco_animal])
                                
                                # For now, return the COCO animal name
                                # In a more advanced implementation, we could classify further
                                animal_name = possible_animals[0] if possible_animals else coco_animal
                                
                                # Get animal features
                                animal_feature = self.animal_features.get(animal_name, f"This is a {animal_name}.")
                                
                                return animal_name, confidence, animal_feature
                
                # No animal detected
                return "Unknown", 0.0, "No animal detected with sufficient confidence."
            
        except Exception as e:
            print(f"Detection error: {e}")
            return "Unknown", 0.0, "Detection error occurred."
    
    def draw_detection(self, frame: np.ndarray, animal_name: str, 
                      confidence: float, position: Tuple[int, int]) -> np.ndarray:
        """
        Draw detection results on a frame.
        
        Args:
            frame: Input frame as numpy array
            animal_name: Detected animal name
            confidence: Detection confidence
            position: Position to draw text (x, y)
            
        Returns:
            Frame with detection results drawn
        """
        # Draw animal name
        cv2.putText(frame, f"Animal: {animal_name}", position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw confidence
        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                   (position[0], position[1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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