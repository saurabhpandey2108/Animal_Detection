#!/usr/bin/env python3
"""
Animal Detection System using YOLO
Real-time animal detection with feature information
Supports both CLI and GUI modes
"""

import cv2
import sys
import os
import time
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import CLI components
from utils.detection import AnimalDetector, initialize_camera, release_camera

# Import GUI components (optional, only if tkinter is available)
try:
    from ui.gui_app import AnimalDetectionGUI
    import tkinter as tk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


def show_animal_info(animal_name: str, confidence: float, features: str) -> None:
    """
    Display information about the detected animal.
    
    Args:
        animal_name: Name of the detected animal
        confidence: Detection confidence level
        features: Features/description of the animal
    """
    print("=" * 50)
    print(f"ANIMAL DETECTED: {animal_name.upper()}")
    print(f"Confidence: {confidence:.2f}")
    print("-" * 30)
    print("FEATURES:")
    print(features)
    print("=" * 50)


def run_cli_mode():
    """Run the animal detection system in CLI mode."""
    print("Animal Detection System using YOLO (CLI Mode)")
    print("Initializing camera...")
    
    try:
        # Initialize camera
        cap = initialize_camera()
        print("Camera initialized successfully!")
        
        # Initialize YOLO animal detector
        detector = AnimalDetector()
        print("YOLO animal detector model loaded!")
        
        print("\nStarting real-time animal detection...")
        print("Press 'q' to quit the application")
        print("Press 'i' to get information about the last detected animal")
        
        last_detected_animal = None
        last_confidence = 0.0
        last_features = ""
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Process every 3rd frame for better performance
            if frame_count % 3 == 0:
                # Detect animal in the frame using YOLO
                animal_name, confidence, features = detector.detect_animal(frame)
                
                # Update last detected animal if confidence is high enough
                if confidence >= 0.3:  # Lower threshold for display
                    last_detected_animal = animal_name
                    last_confidence = confidence
                    last_features = features
                
                # Draw detection on frame
                if last_detected_animal and last_confidence >= 0.3:
                    frame = detector.draw_detection(
                        frame, 
                        last_detected_animal, 
                        last_confidence, 
                        (10, 30)
                    )
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('Animal Detection using YOLO', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i') and last_detected_animal:
                show_animal_info(last_detected_animal, last_confidence, last_features)
        
        # Release everything
        release_camera(cap)
        print("\nCamera released. Application terminated.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_gui_mode():
    """Run the animal detection system in GUI mode."""
    if not GUI_AVAILABLE:
        print("GUI mode is not available. Please install tkinter to use the GUI.")
        print("Falling back to CLI mode...")
        run_cli_mode()
        return
    
    try:
        root = tk.Tk()
        app = AnimalDetectionGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Failed to start GUI: {str(e)}")
        print("Falling back to CLI mode...")
        run_cli_mode()


def main():
    """Main function to run the animal detection system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Animal Detection System using YOLO")
    parser.add_argument("--gui", action="store_true", 
                       help="Run in GUI mode (requires tkinter)")
    args = parser.parse_args()
    
    if args.gui:
        run_gui_mode()
    else:
        run_cli_mode()


if __name__ == "__main__":
    main()