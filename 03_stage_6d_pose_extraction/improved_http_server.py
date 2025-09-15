#!/usr/bin/env python3
"""
Improved HTTP Server for CoppeliaSim
Handles both HTTP requests and file monitoring for camera data
"""

import http.server
import socketserver
import json
import threading
import time
import os
import numpy as np
import cv2
from urllib.parse import urlparse, parse_qs
from ultralytics import YOLO

class ImprovedCoppeliaSimHTTPHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.camera_data = None
        self.last_command = None
        self.yolo_model = None
        super().__init__(*args, **kwargs)
    
    def load_yolo_model(self):
        """Load YOLO model for object detection."""
        if self.yolo_model is None:
            try:
                model_path = "ycb_texture_training/ycb_texture_detector/weights/best.pt"
                if os.path.exists(model_path):
                    self.yolo_model = YOLO(model_path)
                    print(f"✅ Trained YOLO model loaded: {model_path}")
                else:
                    self.yolo_model = YOLO('yolov8n.pt')
                    print("⚠️ Using pre-trained YOLO model")
            except Exception as e:
                print(f"❌ Error loading YOLO model: {e}")
                self.yolo_model = None
    
    def read_camera_data(self, filename):
        """Read camera data and convert to image."""
        try:
            with open(filename, 'rb') as f:
                content = f.read()
            
            data = np.frombuffer(content, dtype=np.uint8)
            
            # Try 512x256 first (most common for CoppeliaSim)
            if len(data) >= 512 * 256 * 3:
                image = data[:512*256*3].reshape((256, 512, 3))
                return image
            
            # Try other common sizes
            sizes = [(640, 480, 3), (800, 600, 3), (1024, 768, 3), (256, 256, 3)]
            for width, height, channels in sizes:
                expected_size = width * height * channels
                if len(data) >= expected_size:
                    image = data[:expected_size].reshape((height, width, channels))
                    return image
            
            return None
        except Exception as e:
            return None
    
    def detect_objects(self, image):
        """Detect objects using YOLO model."""
        if self.yolo_model is None or image is None:
            return []
        
        try:
            results = self.yolo_model(image, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        if hasattr(result, 'names') and cls in result.names:
                            class_name = result.names[cls]
                        else:
                            class_name = f"Class_{cls}"
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class': class_name,
                            'class_id': int(cls)
                        })
            
            return detections
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
            return []
    
    def do_GET(self):
        """Handle GET requests from CoppeliaSim."""
        try:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            query_params = parse_qs(parsed_url.query)
            
            print(f"📥 GET request: {path}")
            
            if path == "/status":
                # Return server status
                response = {
                    "status": "running",
                    "timestamp": time.time(),
                    "camera_data_available": self.camera_data is not None
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            elif path == "/camera":
                # Return camera data
                if self.camera_data:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/octet-stream')
                    self.end_headers()
                    self.wfile.write(self.camera_data)
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "No camera data available"}).encode())
                    
            elif path == "/detect":
                # Run object detection on current camera data
                if self.camera_data:
                    # Convert camera data to image
                    data = np.frombuffer(self.camera_data, dtype=np.uint8)
                    if len(data) >= 512 * 256 * 3:
                        image = data[:512*256*3].reshape((256, 512, 3))
                        
                        # Load YOLO model if not loaded
                        self.load_yolo_model()
                        
                        # Run detection
                        detections = self.detect_objects(image)
                        
                        response = {
                            "status": "success",
                            "detections": detections,
                            "timestamp": time.time()
                        }
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode())
                    else:
                        self.send_response(400)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "Invalid camera data"}).encode())
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "No camera data available"}).encode())
                    
            else:
                # Return 404 for unknown paths
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Not found"}).encode())
                
        except Exception as e:
            print(f"❌ GET request error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def do_POST(self):
        """Handle POST requests from CoppeliaSim."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            
            print(f"📥 POST request: {path}")
            
            if path == "/camera":
                # Receive camera data
                self.camera_data = post_data
                print(f"✅ Received camera data: {len(post_data)} bytes")
                
                # Process the camera data
                self.process_camera_data(post_data)
                
                response = {"status": "success", "bytes_received": len(post_data)}
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            else:
                # Return 404 for unknown paths
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Not found"}).encode())
                
        except Exception as e:
            print(f"❌ POST request error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def process_camera_data(self, camera_data):
        """Process camera data and run object detection."""
        try:
            # Convert to image
            data = np.frombuffer(camera_data, dtype=np.uint8)
            if len(data) >= 512 * 256 * 3:
                image = data[:512*256*3].reshape((256, 512, 3))
                
                # Save image
                cv2.imwrite("http_camera_capture.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                print("💾 Camera image saved as: http_camera_capture.jpg")
                
                # Load YOLO model if not loaded
                self.load_yolo_model()
                
                # Run object detection
                detections = self.detect_objects(image)
                
                if detections:
                    print(f"🔍 Detected {len(detections)} objects:")
                    for det in detections:
                        print(f"   - {det['class']} (conf: {det['confidence']:.3f})")
                    
                    # Save detection results
                    detection_data = {
                        "timestamp": time.time(),
                        "detections": detections,
                        "image_file": "http_camera_capture.jpg"
                    }
                    
                    with open("http_detection_results.json", "w") as f:
                        json.dump(detection_data, f, indent=2)
                    
                    print("💾 Detection results saved as: http_detection_results.json")
                else:
                    print("❌ No objects detected")
                    
        except Exception as e:
            print(f"❌ Error processing camera data: {e}")

class ImprovedCoppeliaSimHTTPServer:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False
        self.last_file_time = 0
        
    def start_server(self):
        """Start the HTTP server in a separate thread."""
        try:
            # Create server
            handler = ImprovedCoppeliaSimHTTPHandler
            self.server = socketserver.TCPServer((self.host, self.port), handler)
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.running = True
            print(f"✅ Improved HTTP server started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start HTTP server: {e}")
            return False
    
    def monitor_camera_files(self):
        """Monitor camera data files for changes."""
        try:
            rgb_file = "current_rgb.txt"
            
            if os.path.exists(rgb_file):
                file_time = os.path.getmtime(rgb_file)
                
                if file_time > self.last_file_time:
                    print(f"📸 New camera data detected at {time.ctime(file_time)}")
                    
                    # Read the file
                    with open(rgb_file, 'rb') as f:
                        camera_data = f.read()
                    
                    # Process the data
                    self.process_camera_data(camera_data)
                    
                    self.last_file_time = file_time
                    
        except Exception as e:
            print(f"❌ Error monitoring camera files: {e}")
    
    def stop_server(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            print("🛑 HTTP server stopped")

def main():
    """Test improved HTTP server communication."""
    print("🌐 Starting Improved HTTP Server for CoppeliaSim")
    print("=" * 60)
    
    # Start HTTP server
    server = ImprovedCoppeliaSimHTTPServer()
    if server.start_server():
        print("✅ HTTP server is running")
        print("\n💡 Instructions:")
        print("   1. Copy the Lua script from simple_coppelia_http_client.lua")
        print("   2. Paste it into CoppeliaSim console")
        print("   3. The script will capture camera data and send it to the server")
        print("   4. Object detection will run automatically")
        print("   5. Press Ctrl+C to stop the server")
        
        try:
            while True:
                time.sleep(1)
                
                # Monitor camera files
                server.monitor_camera_files()
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
        finally:
            server.stop_server()
    else:
        print("❌ Failed to start HTTP server")

if __name__ == "__main__":
    main()
