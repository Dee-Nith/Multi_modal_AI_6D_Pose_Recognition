#!/usr/bin/env python3
"""
Main Entry Point - AI 6D Pose Recognition Robotic Grasping System
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clients import CoppeliaSimZMQClient

def test_connection():
    """Test CoppeliaSim connection."""
    print("🤖 Testing CoppeliaSim Connection...")
    try:
        with CoppeliaSimZMQClient() as client:
            if client.test_connection():
                print("✅ Connection successful!")
                return True
            else:
                print("❌ Connection failed!")
                return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="AI 6D Pose Recognition - Robotic Grasping System")
    parser.add_argument("--test", action="store_true", help="Test CoppeliaSim connection")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--grasp", action="store_true", help="Run grasping pipeline")
    
    args = parser.parse_args()
    
    if args.test:
        test_connection()
    elif args.train:
        print("🚧 Training functionality coming soon...")
        print("Use the training scripts in src/training/")
    elif args.evaluate:
        print("🚧 Evaluation functionality coming soon...")
        print("Use the evaluation scripts in src/evaluation/")
    elif args.grasp:
        print("🚧 Grasping pipeline coming soon...")
        print("Use the grasping scripts in src/")
    else:
        print("🤖 AI 6D Pose Recognition - Robotic Grasping System")
        print("=" * 50)
        print("Available commands:")
        print("  python main.py --test     # Test CoppeliaSim connection")
        print("  python main.py --train    # Train models")
        print("  python main.py --evaluate # Evaluate models")
        print("  python main.py --grasp    # Run grasping pipeline")
        print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
