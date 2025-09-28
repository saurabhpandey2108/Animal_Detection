"""
GUI Application for Animal Detection using YOLO
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import time
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from animal_detection.utils.detection import AnimalDetector, initialize_camera, release_camera


class AnimalDetectionGUI:
    """GUI application for real-time animal detection."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Animal Detection System")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Application state
        self.is_detecting = False
        self.cap = None
        self.detector = None
        self.last_detected_animal = None
        self.last_confidence = 0.0
        self.last_features = ""
        
        # Create UI elements
        self.create_widgets()
        
        # Initialize detection components
        self.init_detection_components()
        
    def create_widgets(self):
        """Create all UI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Animal Detection System", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", 
                                      command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", 
                                     command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.info_button = ttk.Button(button_frame, text="Animal Info", 
                                     command=self.show_animal_info, state=tk.DISABLED)
        self.info_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.exit_button = ttk.Button(button_frame, text="Exit", 
                                     command=self.root.quit)
        self.exit_button.pack(side=tk.RIGHT)
        
        # Video display frame
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas for video display
        self.video_canvas = tk.Canvas(video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to start detection")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Detection info panel
        info_frame = ttk.LabelFrame(main_frame, text="Detection Information")
        info_frame.pack(fill=tk.X)
        
        # Animal name
        name_frame = ttk.Frame(info_frame)
        name_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(name_frame, text="Animal:").pack(side=tk.LEFT)
        self.animal_var = tk.StringVar(value="None")
        ttk.Label(name_frame, textvariable=self.animal_var, 
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(5, 0))
        
        # Confidence
        conf_frame = ttk.Frame(info_frame)
        conf_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT)
        self.confidence_var = tk.StringVar(value="0.00")
        ttk.Label(conf_frame, textvariable=self.confidence_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # FPS
        fps_frame = ttk.Frame(info_frame)
        fps_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="0.0")
        ttk.Label(fps_frame, textvariable=self.fps_var).pack(side=tk.LEFT, padx=(5, 0))
        
    def init_detection_components(self):
        """Initialize detection components."""
        try:
            self.detector = AnimalDetector()
            self.status_var.set("Detector initialized. Click 'Start Detection' to begin.")
        except Exception as e:
            messagebox.showerror("Initialization Error", 
                               f"Failed to initialize detector: {str(e)}")
            self.status_var.set("Detector initialization failed")
    
    def start_detection(self):
        """Start the animal detection process."""
        if self.is_detecting:
            return
            
        try:
            # Initialize camera
            self.cap = initialize_camera()
            self.is_detecting = True
            
            # Update UI state
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.info_button.config(state=tk.DISABLED)
            self.status_var.set("Detection started. Processing video feed...")
            
            # Reset detection info
            self.animal_var.set("None")
            self.confidence_var.set("0.00")
            self.last_detected_animal = None
            self.last_confidence = 0.0
            self.last_features = ""
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.run_detection)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Camera Error", 
                               f"Failed to initialize camera: {str(e)}")
            self.status_var.set("Camera initialization failed")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def stop_detection(self):
        """Stop the animal detection process."""
        self.is_detecting = False
        
        # Release camera
        if self.cap:
            release_camera(self.cap)
            self.cap = None
        
        # Update UI state
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.info_button.config(state=tk.DISABLED)
        self.status_var.set("Detection stopped")
        
        # Clear video canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            self.video_canvas.winfo_width() // 2, 
            self.video_canvas.winfo_height() // 2,
            text="Video feed stopped", 
            fill="white", 
            font=("Arial", 12)
        )
    
    def run_detection(self):
        """Run the detection loop in a separate thread."""
        frame_count = 0
        start_time = time.time()
        
        while self.is_detecting and self.cap:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    self.status_var.set("Failed to grab frame")
                    break
                
                frame_count += 1
                
                # Calculate FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                self.fps_var.set(f"{fps:.1f}")
                
                # Process every 3rd frame for performance
                if frame_count % 3 == 0 and self.detector:
                    # Detect animal
                    animal_name, confidence, features = self.detector.detect_animal(frame)
                    
                    # Update detection info if confidence is sufficient
                    if confidence >= 0.3:
                        self.last_detected_animal = animal_name
                        self.last_confidence = confidence
                        self.last_features = features
                        
                        # Update UI in main thread
                        self.root.after(0, self.update_detection_info)
                
                # Display frame
                self.display_frame(frame)
                
            except Exception as e:
                self.status_var.set(f"Detection error: {str(e)}")
                break
        
        # Clean up when stopping
        if self.cap:
            release_camera(self.cap)
            self.cap = None
    
    def update_detection_info(self):
        """Update the detection information display."""
        if self.last_detected_animal:
            self.animal_var.set(self.last_detected_animal)
            self.confidence_var.set(f"{self.last_confidence:.2f}")
            self.info_button.config(state=tk.NORMAL)
        else:
            self.animal_var.set("None")
            self.confidence_var.set("0.00")
            self.info_button.config(state=tk.DISABLED)
    
    def display_frame(self, frame):
        """Display a frame on the canvas."""
        # Draw detection info on frame
        if self.last_detected_animal and self.last_confidence >= 0.3:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Draw text
            cv2.putText(rgb_frame, f"Animal: {self.last_detected_animal}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(rgb_frame, f"Confidence: {self.last_confidence:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize to fit canvas while maintaining aspect ratio
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Check if canvas has valid size
            # Calculate aspect ratio
            img_width, img_height = pil_image.size
            aspect_ratio = img_width / img_height
            
            # Calculate new dimensions
            if canvas_width / canvas_height > aspect_ratio:
                new_height = canvas_height
                new_width = int(canvas_height * aspect_ratio)
            else:
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)
            
            # Resize image
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Display on canvas
        self.video_canvas.delete("all")
        x = (self.video_canvas.winfo_width() - pil_image.width) // 2
        y = (self.video_canvas.winfo_height() - pil_image.height) // 2
        self.video_canvas.create_image(x, y, anchor=tk.NW, image=photo)
        self.video_canvas.image = photo  # Keep a reference
    
    def show_animal_info(self):
        """Show detailed information about the last detected animal."""
        if not self.last_detected_animal:
            messagebox.showinfo("Animal Information", "No animal detected yet.")
            return
            
        # Create info window
        info_window = tk.Toplevel(self.root)
        info_window.title(f"Information: {self.last_detected_animal}")
        info_window.geometry("500x400")
        info_window.minsize(400, 300)
        
        # Animal name
        name_label = ttk.Label(info_window, text=self.last_detected_animal, 
                              font=("Arial", 14, "bold"))
        name_label.pack(pady=(10, 5))
        
        # Confidence
        conf_label = ttk.Label(info_window, 
                              text=f"Detection Confidence: {self.last_confidence:.2f}")
        conf_label.pack(pady=(0, 10))
        
        # Features text area
        text_frame = ttk.LabelFrame(info_window, text="Features")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        text_area = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, 
                                             width=50, height=15)
        text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        text_area.insert(tk.END, self.last_features)
        text_area.config(state=tk.DISABLED)
        
        # Close button
        close_button = ttk.Button(info_window, text="Close", 
                                 command=info_window.destroy)
        close_button.pack(pady=(0, 10))
    
    def on_closing(self):
        """Handle application closing."""
        self.is_detecting = False
        if self.cap:
            release_camera(self.cap)
        self.root.destroy()


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = AnimalDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()