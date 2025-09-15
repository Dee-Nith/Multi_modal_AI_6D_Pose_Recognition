#!/usr/bin/env python3
"""
View Batch 6D Pose Estimation Results
Interactive viewer for comprehensive batch processing results
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json
import numpy as np

def view_batch_results():
    """View and display all batch processing results"""
    print("🎯 Viewing Batch 6D Pose Estimation Results")
    print("=" * 45)
    
    results_dir = Path("batch_6d_pose_results")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # 1. Show summary visualization
    summary_image = results_dir / "batch_6d_pose_summary.png"
    if summary_image.exists():
        print(f"\n📊 Loading batch summary visualization...")
        
        # Load and display the summary image
        img = mpimg.imread(str(summary_image))
        
        plt.figure(figsize=(20, 12))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Batch 6D Pose Estimation Results - Complete Summary', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        print(f"✅ Summary visualization displayed")
    
    # 2. Load and display key statistics
    json_file = results_dir / "batch_6d_pose_results.json"
    if json_file.exists():
        print(f"\n📋 Loading detailed results...")
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
            
            # Calculate overall statistics
            total_captures = len(results)
            total_objects = sum(r['total_objects'] for r in results)
            total_poses = sum(r['successful_poses'] for r in results)
            success_rate = (total_poses / total_objects * 100) if total_objects > 0 else 0
            
            print(f"✅ Detailed results loaded:")
            print(f"   📸 Total Captures: {total_captures}")
            print(f"   🎯 Total Objects: {total_objects}")
            print(f"   📐 Total 6D Poses: {total_poses}")
            print(f"   📈 Success Rate: {success_rate:.1f}%")
            
            # Show per-capture breakdown
            print(f"\n📊 Per-Capture Breakdown:")
            for result in results[:10]:  # Show first 10 captures
                capture_num = result['capture_number']
                objects = result['total_objects']
                poses = result['successful_poses']
                rate = (poses/objects*100) if objects > 0 else 0
                print(f"   Capture {capture_num}: {objects} objects → {poses} poses ({rate:.0f}%)")
            
            if len(results) > 10:
                print(f"   ... and {len(results)-10} more captures")
                
        except Exception as e:
            print(f"   ⚠️ Could not load JSON: {e}")
    
    # 3. Show text summary sample
    text_file = results_dir / "batch_summary.txt"
    if text_file.exists():
        print(f"\n📄 Sample from text summary:")
        with open(text_file, 'r') as f:
            lines = f.readlines()
            # Show first 30 lines
            for line in lines[:30]:
                print(f"   {line.rstrip()}")
            if len(lines) > 30:
                print(f"   ... ({len(lines)-30} more lines)")
    
    # 4. Create a simple statistics plot
    if json_file.exists():
        try:
            print(f"\n📊 Creating statistics visualization...")
            
            # Extract data for visualization
            capture_numbers = []
            objects_per_capture = []
            poses_per_capture = []
            
            for result in results:
                capture_numbers.append(int(result['capture_number']))
                objects_per_capture.append(result['total_objects'])
                poses_per_capture.append(result['successful_poses'])
            
            # Sort by capture number
            sorted_data = sorted(zip(capture_numbers, objects_per_capture, poses_per_capture))
            capture_numbers, objects_per_capture, poses_per_capture = zip(*sorted_data)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Objects detected per capture
            ax1.bar(capture_numbers, objects_per_capture, color='lightblue', alpha=0.7, edgecolor='navy')
            ax1.set_title('Objects Detected per Capture', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Capture Number')
            ax1.set_ylabel('Number of Objects')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(objects_per_capture):
                ax1.text(capture_numbers[i], v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
            
            # 6D poses estimated per capture
            ax2.bar(capture_numbers, poses_per_capture, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
            ax2.set_title('6D Poses Successfully Estimated per Capture', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Capture Number')
            ax2.set_ylabel('Number of 6D Poses')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(poses_per_capture):
                ax2.text(capture_numbers[i], v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            print(f"✅ Statistics visualization displayed")
            
        except Exception as e:
            print(f"   ⚠️ Could not create statistics plot: {e}")
    
    print(f"\n🎉 All available results have been displayed!")
    print(f"📁 Full results available in: {results_dir}")
    print(f"📄 Files available:")
    for file in results_dir.glob("*"):
        size_mb = file.stat().st_size / (1024*1024)
        print(f"   - {file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    view_batch_results()




