#!/usr/bin/env python3
"""
GPU Training Setup Script
Checks GPU availability and installs required dependencies
"""

import subprocess
import sys
import platform

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required for YOLOv8 training")
        return False
    
    print("✅ Python version compatible")
    return True

def check_pip():
    """Check pip availability"""
    print("\n📦 Checking pip...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ pip available")
            return True
        else:
            print("❌ pip not available")
            return False
    except Exception as e:
        print(f"❌ Error checking pip: {e}")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📥 Installing requirements...")
    
    requirements = [
        "torch",
        "torchvision", 
        "ultralytics",
        "opencv-python",
        "numpy",
        "Pillow",
        "PyYAML",
        "matplotlib",
        "seaborn",
        "pandas",
        "tqdm",
        "scipy"
    ]
    
    for package in requirements:
        print(f"   Installing {package}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ {package} installed")
            else:
                print(f"   ❌ Failed to install {package}")
                print(f"      Error: {result.stderr}")
        except Exception as e:
            print(f"   ❌ Error installing {package}: {e}")

def check_gpu_support():
    """Check GPU and CUDA support"""
    print("\n🚀 Checking GPU support...")
    
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("   ✅ CUDA available")
            print(f"   🔢 CUDA version: {torch.version.cuda}")
            print(f"   🖥️  GPU device: {torch.cuda.get_device_name(0)}")
            print(f"   💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test GPU computation
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.mm(x, x)
                print("   ✅ GPU computation test passed")
                return True
            except Exception as e:
                print(f"   ❌ GPU computation test failed: {e}")
                return False
        else:
            print("   ❌ CUDA not available")
            print("   💡 Install PyTorch with CUDA support:")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False
            
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_ultralytics():
    """Check Ultralytics installation"""
    print("\n🔍 Checking Ultralytics...")
    
    try:
        import ultralytics
        print(f"   ✅ Ultralytics version: {ultralytics.__version__}")
        
        # Check YOLO models
        from ultralytics import YOLO
        print("   ✅ YOLO models available")
        return True
        
    except ImportError:
        print("   ❌ Ultralytics not installed")
        return False

def main():
    """Main setup function"""
    print("🎯 GPU Training Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        print("❌ pip not available. Please install pip first.")
        sys.exit(1)
    
    # Install requirements
    install_requirements()
    
    # Check GPU support
    gpu_available = check_gpu_support()
    
    # Check Ultralytics
    ultralytics_available = check_ultralytics()
    
    print("\n" + "=" * 50)
    if gpu_available and ultralytics_available:
        print("🎉 Setup completed successfully!")
        print("   🚀 GPU training ready")
        print("   🔍 Run: python train_instance_segmentation.py")
    else:
        print("⚠️  Setup completed with issues:")
        if not gpu_available:
            print("   ❌ GPU not available for training")
        if not ultralytics_available:
            print("   ❌ Ultralytics not properly installed")
        print("   💡 Check the errors above and reinstall if needed")

if __name__ == "__main__":
    main()




