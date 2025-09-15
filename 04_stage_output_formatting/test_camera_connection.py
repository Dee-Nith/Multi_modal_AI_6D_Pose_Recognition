#!/usr/bin/env python3
"""
Simple test to check CoppeliaSim camera connection
"""

import requests
import time

def test_coppelia_connection():
    """Test basic connection to CoppeliaSim."""
    print("🔍 Testing CoppeliaSim connection...")
    
    # Try different ports
    ports = [23000, 23001, 23002, 23003]
    
    for port in ports:
        try:
            print(f"  Trying port {port}...")
            response = requests.get(f"http://localhost:{port}", timeout=2)
            print(f"  ✅ Port {port} responded: {response.status_code}")
            
            # Try to get simulation info
            try:
                sim_response = requests.get(f"http://localhost:{port}/api/v1/sim", timeout=2)
                print(f"  📊 Simulation API: {sim_response.status_code}")
                if sim_response.status_code == 200:
                    print(f"  📄 Response: {sim_response.text[:100]}...")
            except:
                print(f"  ❌ No simulation API on port {port}")
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Port {port}: {e}")
    
    print("\n💡 If no ports work, please:")
    print("   1. Open CoppeliaSim")
    print("   2. Go to Tools > Developer > Web Server")
    print("   3. Enable the web server on port 23000")
    print("   4. Restart CoppeliaSim if needed")

if __name__ == "__main__":
    test_coppelia_connection()







