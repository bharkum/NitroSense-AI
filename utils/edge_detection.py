
import cv2
import numpy as np

class EdgeDetector:
    def __init__(self, threshold1=50, threshold2=150):
        """
        Initialize Edge Detector
        
        Args:
            threshold1: First threshold for Canny edge detection
            threshold2: Second threshold for Canny edge detection
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2
    
    def detect_edges(self, image):
        """
        Detect edges and highlight them in red on white background
        
        Args:
            image: Input image (RGB numpy array)
            
        Returns:
            edge_image: Image with red edges on white background
            edges: Binary edge mask
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.threshold1, self.threshold2)
        
        # Create white background with red edges
        edge_image = np.ones((edges.shape[0], edges.shape[1], 3), dtype=np.uint8) * 255
        edge_image[edges != 0] = [255, 0, 0]  # Red edges
        
        return edge_image, edges
    
    def adjust_thresholds(self, threshold1, threshold2):
        """
        Adjust the edge detection thresholds
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        print(f"Thresholds updated: threshold1={threshold1}, threshold2={threshold2}")
