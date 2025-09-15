#!/usr/bin/env python3
"""
Test Robot Command Generation
Simple script to test robot command file generation without running the full system
"""

import json
import os
import time

def test_robot_command_generation():
    """Test the robot command generation system"""
    print("🧪 Testing Robot Command Generation System")
    print("=" * 50)
    
    # Test data (similar to what the real system would generate)
    test_sequence = [
        {
            'type': 'pick',
            'object_name': 'master_chef_can',
            'position': [-0.625, -0.275, 0.750],
            'rotation': [0, 0, 45]
        },
        {
            'type': 'place',
            'object_name': 'master_chef_can',
            'position': [0.100, -0.625, 0.700],
            'rotation': [0, 0, 0]
        },
        {
            'type': 'pick',
            'object_name': 'cracker_box',
            'position': [-0.625, -0.100, 0.825],
            'rotation': [0, 0, 0]
        },
        {
            'type': 'place',
            'object_name': 'cracker_box',
            'position': [0.100, -0.625, 0.700],
            'rotation': [0, 0, 0]
        }
    ]
    
    # Robot base and place conveyor positions
    robot_base = {'x': -0.625, 'y': 0.075, 'z': 0.700}
    place_conveyor = {'x': 0.100, 'y': -0.625, 'z': 0.700}
    
    print("📋 Test Sequence:")
    for i, op in enumerate(test_sequence):
        print(f"  {i+1}. {op['type'].upper()}: {op['object_name']} at {op['position']}")
    
    print("\n🤖 Generating robot commands...")
    
    # Create command directory
    command_dir = "/tmp/robot_commands"
    os.makedirs(command_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = int(time.time())
    command_file = f"{command_dir}/test_robot_commands_{timestamp}.txt"
    
    # Generate simple command file
    with open(command_file, 'w') as f:
        f.write(f"# Test Robot Pick and Place Commands - Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total Operations: {len(test_sequence)}\n")
        f.write(f"# Robot Base: {robot_base}\n")
        f.write(f"# Place Conveyor: {place_conveyor}\n\n")
        
        for i, operation in enumerate(test_sequence):
            op_type = operation['type']
            object_name = operation['object_name']
            position = operation['position']
            rotation = operation['rotation']
            
            f.write(f"# Operation {i+1}: {op_type.upper()} {object_name}\n")
            
            if op_type == 'pick':
                # Pick sequence
                f.write(f"PICK_START {object_name}\n")
                f.write(f"MOVE_APPROACH {position[0]:.6f} {position[1]:.6f} {position[2]+0.1:.6f} {rotation[0]:.1f} {rotation[1]:.1f} {rotation[2]:.1f}\n")
                f.write(f"GRIPPER_OPEN\n")
                f.write(f"MOVE_PICK {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} {rotation[0]:.1f} {rotation[1]:.1f} {rotation[2]:.1f}\n")
                f.write(f"GRIPPER_CLOSE\n")
                f.write(f"MOVE_LIFT {position[0]:.6f} {position[1]:.6f} {position[2]+0.15:.6f} {rotation[0]:.1f} {rotation[1]:.1f} {rotation[2]:.1f}\n")
                f.write(f"PICK_END {object_name}\n\n")
                
            elif op_type == 'place':
                # Place sequence
                f.write(f"PLACE_START {object_name}\n")
                f.write(f"MOVE_PLACE_APPROACH {position[0]:.6f} {position[1]:.6f} {position[2]+0.1:.6f} 0.0 0.0 0.0\n")
                f.write(f"MOVE_PLACE {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} 0.0 0.0 0.0\n")
                f.write(f"GRIPPER_OPEN\n")
                f.write(f"MOVE_PLACE_APPROACH {position[0]:.6f} {position[1]:.6f} {position[2]+0.15:.6f} 0.0 0.0 0.0\n")
                f.write(f"PLACE_END {object_name}\n\n")
        
        # Add return home command
        f.write(f"# Final: Return to home position\n")
        f.write(f"RETURN_HOME {robot_base['x']:.6f} {robot_base['y']:.6f} {robot_base['z']+0.3:.6f} 0.0 0.0 0.0\n")
        f.write(f"SYSTEM_READY\n")
    
    print(f"✅ Test robot commands generated: {command_file}")
    
    # Show the contents of the file
    print("\n📄 Command File Contents:")
    print("-" * 40)
    with open(command_file, 'r') as f:
        content = f.read()
        print(content)
    
    print("\n🎯 Next Steps:")
    print("1. Copy the command file to CoppeliaSim")
    print("2. Run the 'coppelia_robot_controller.lua' script")
    print("3. Watch the robot execute the commands!")
    
    return command_file

if __name__ == "__main__":
    test_robot_command_generation()




