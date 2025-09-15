#!/usr/bin/env python3
"""
Live CoppeliaSim Connection
Direct connection to CoppeliaSim for real-time camera feed
"""

import zmq
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import struct

class LiveCoppeliaConnection:
    def __init__(self, port=23000):
        self.port = port
        self.context = zmq.Context()
        self.socket = None
        self.connected = False
        self.yolo_model = None
        
        # Load YOLO model
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load YOLO model: {e}")
    
    def connect(self):
        """Connect to CoppeliaSim via ZeroMQ."""
        try:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://localhost:{self.port}")
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
            self.connected = True
            print(f"✅ Connected to CoppeliaSim on port {self.port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to CoppeliaSim: {e}")
            return False
    
    def send_command(self, command):
        """Send a command to CoppeliaSim."""
        if not self.connected:
            return None
        
        try:
            # Pack command as bytes
            cmd_bytes = struct.pack('<I', len(command)) + command.encode('utf-8')
            self.socket.send(cmd_bytes)
            
            # Receive response
            response = self.socket.recv()
            return response
        except Exception as e:
            print(f"❌ Communication error: {e}")
            return None
    
    def get_camera_handles(self):
        """Get handles for RGB and depth cameras."""
        print("🔍 Getting camera handles...")
        
        # Get RGB camera handle
        rgb_response = self.send_command('sim.getObjectHandle("./sensorRGB")')
        if rgb_response:
            rgb_handle = struct.unpack('<I', rgb_response[:4])[0]
            print(f"✅ RGB camera handle: {rgb_handle}")
        else:
            print("❌ Failed to get RGB camera handle")
            return None, None
        
        # Get depth camera handle
        depth_response = self.send_command('sim.getObjectHandle("./sensorDepth")')
        if depth_response:
            depth_handle = struct.unpack('<I', depth_response[:4])[0]
            print(f"✅ Depth camera handle: {depth_handle}")
        else:
            print("❌ Failed to get depth camera handle")
            return None, None
        
        return rgb_handle, depth_handle
    
    def get_camera_image(self, handle):
        """Get image from camera handle."""
        if handle is None:
            return None
        
        # Get image data
        cmd = f'sim.getVisionSensorImg({handle})'
        response = self.send_command(cmd)
        
        if response and len(response) > 4:
            # Parse image data
            data_size = struct.unpack('<I', response[:4])[0]
            if data_size > 0:
                # Convert response to image array
                # This is a simplified approach - actual implementation depends on CoppeliaSim's data format
                try:
                    # Assuming response contains raw image data
                    img_data = response[4:]
                    # Convert to numpy array (this needs to be adapted based on actual data format)
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    return img_array
                except Exception as e:
                    print(f"❌ Error parsing image data: {e}")
        
        return None
    
    def get_camera_depth(self, handle):
        """Get depth data from camera handle."""
        if handle is None:
            return None
        
        # Get depth data
        cmd = f'sim.getVisionSensorDepth({handle})'
        response = self.send_command(cmd)
        
        if response and len(response) > 4:
            # Parse depth data
            data_size = struct.unpack('<I', response[:4])[0]
            if data_size > 0:
                try:
                    # Convert response to depth array
                    depth_data = response[4:]
                    depth_array = np.frombuffer(depth_data, dtype=np.float32)
                    return depth_array
                except Exception as e:
                    print(f"❌ Error parsing depth data: {e}")
        
        return None
    
    def detect_objects(self, image):
        """Detect objects using YOLO."""
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
                        class_name = result.names[cls]
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': class_name,
                            'class_id': cls
                        })
            
            return detections
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
            return []
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image."""
        if image is None:
            return image
        
        img_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Draw bounding box
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img_copy, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_copy
    
    def start_live_feed(self):
        """Start live camera feed with object detection."""
        print("🎥 Starting live camera feed...")
        print("📋 Instructions:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save current frame")
        print("   - Press 'd' to toggle object detection")
        
        if not self.connect():
            print("❌ Could not connect to CoppeliaSim")
            return
        
        # Get camera handles
        rgb_handle, depth_handle = self.get_camera_handles()
        if rgb_handle is None or depth_handle is None:
            print("❌ Could not get camera handles")
            return
        
        show_detections = True
        frame_count = 0
        
        while True:
            try:
                # Get camera images
                rgb_data = self.get_camera_image(rgb_handle)
                depth_data = self.get_camera_depth(depth_handle)
                
                if rgb_data is not None:
                    # Reshape RGB data (assuming 512x256 resolution)
                    try:
                        rgb_image = rgb_data.reshape((256, 512, 3))
                        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                        
                        # Create display image
                        display_img = rgb_image.copy()
                        
                        # Perform object detection if enabled
                        if show_detections:
                            detections = self.detect_objects(rgb_image)
                            display_img = self.draw_detections(display_img, detections)
                            
                            # Print detection info
                            if detections:
                                print(f"🔍 Frame {frame_count}: Detected {len(detections)} objects")
                                for det in detections:
                                    print(f"   - {det['class']} (conf: {det['confidence']:.2f})")
                        
                        # Display the image
                        cv2.imshow('CoppeliaSim Live Camera Feed', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                        
                        # Handle key presses
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            # Save current frame
                            timestamp = int(time.time())
                            cv2.imwrite(f'live_frame_{timestamp}.jpg', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                            print(f"💾 Saved frame: live_frame_{timestamp}.jpg")
                        elif key == ord('d'):
                            show_detections = not show_detections
                            print(f"🔍 Object detection: {'ON' if show_detections else 'OFF'}")
                        
                        frame_count += 1
                        
                    except Exception as e:
                        print(f"❌ Error processing image: {e}")
                
                # Small delay
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in live feed: {e}")
                break
        
        cv2.destroyAllWindows()
        self.disconnect()
    
    def disconnect(self):
        """Disconnect from CoppeliaSim."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("🔌 Disconnected from CoppeliaSim")

def main():
    """Main function."""
    print("🤖 Live CoppeliaSim Camera Connection")
    print("=" * 40)
    
    # Create connection
    connection = LiveCoppeliaConnection()
    
    try:
        connection.start_live_feed()
    except KeyboardInterrupt:
        print("\n🛑 Stopping live feed...")
    finally:
        connection.disconnect()

if __name__ == "__main__":
    main()







