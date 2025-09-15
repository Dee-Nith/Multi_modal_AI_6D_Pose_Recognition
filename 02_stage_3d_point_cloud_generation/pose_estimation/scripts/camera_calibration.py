#!/usr/bin/env python3
"""
Camera Calibration for CoppeliaSim 6D Pose Estimation
Calibrates the Kinect camera in CoppeliaSim
"""

import cv2
import numpy as np
import json
from pathlib import Path

class CoppeliaSimCameraCalibration:
    def __init__(self, chessboard_size=(9, 6), square_size=0.025):
        """
        Initialize camera calibration
        
        Args:
            chessboard_size: (width, height) of chessboard corners
            square_size: Size of chessboard squares in meters
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
        # Camera matrix and distortion coefficients
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def add_chessboard_image(self, image):
        """
        Add a chessboard image for calibration
        
        Args:
            image: Input image with chessboard pattern
            
        Returns:
            bool: True if chessboard was found and added
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # Refine corner detection
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Add points
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners2)
            
            return True
        return False
    
    def calibrate(self, image_size):
        """
        Calibrate camera using collected points
        
        Args:
            image_size: (width, height) of images
            
        Returns:
            bool: True if calibration successful
        """
        if len(self.objpoints) < 5:
            print("⚠️ Need at least 5 chessboard images for calibration")
            return False
            
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None
        )
        
        if ret:
            print(f"✅ Camera calibrated successfully with {len(self.objpoints)} images")
            return True
        else:
            print("❌ Camera calibration failed")
            return False
    
    def get_calibration_data(self):
        """
        Get calibration data as dictionary
        
        Returns:
            dict: Camera matrix and distortion coefficients
        """
        if self.camera_matrix is None:
            return None
            
        return {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'image_width': 640,  # CoppeliaSim Kinect resolution
            'image_height': 480
        }
    
    def save_calibration(self, filepath):
        """
        Save calibration data to JSON file
        
        Args:
            filepath: Path to save calibration file
        """
        data = self.get_calibration_data()
        if data:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Calibration saved to {filepath}")
        else:
            print("❌ No calibration data to save")
    
    def load_calibration(self, filepath):
        """
        Load calibration data from JSON file
        
        Args:
            filepath: Path to calibration file
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.camera_matrix = np.array(data['camera_matrix'])
            self.dist_coeffs = np.array(data['dist_coeffs'])
            print(f"✅ Calibration loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            return False

def create_default_coppelia_calibration():
    """
    Create default calibration for CoppeliaSim Kinect camera
    Uses typical Kinect parameters
    """
    # Typical Kinect camera parameters (640x480)
    camera_matrix = np.array([
        [525.0, 0.0, 320.0],
        [0.0, 525.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Minimal distortion (Kinect has low distortion)
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    calibration = CoppeliaSimCameraCalibration()
    calibration.camera_matrix = camera_matrix
    calibration.dist_coeffs = dist_coeffs
    
    return calibration

if __name__ == "__main__":
    # Create default CoppeliaSim calibration
    calibration = create_default_coppelia_calibration()
    
    # Save calibration
    calibration_path = Path("../calibration/coppelia_camera_calibration.json")
    calibration_path.parent.mkdir(exist_ok=True)
    calibration.save_calibration(calibration_path)
    
    print("🎯 CoppeliaSim camera calibration ready!")
    print("📁 Calibration saved for 6D pose estimation")




