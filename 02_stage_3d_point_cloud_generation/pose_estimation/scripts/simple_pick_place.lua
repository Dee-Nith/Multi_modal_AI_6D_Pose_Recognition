-- Simple Pick and Place Script
-- Just pick Master Chef Can and place it

function sysCall_init()
    -- Get robot handles
    robot = sim.getObjectHandle("UR5")
    
    -- Get joint handles
    joint1 = sim.getObjectHandle("UR5_joint1")
    joint2 = sim.getObjectHandle("UR5_joint2")
    joint3 = sim.getObjectHandle("UR5_joint3")
    joint4 = sim.getObjectHandle("UR5_joint4")
    joint5 = sim.getObjectHandle("UR5_joint5")
    joint6 = sim.getObjectHandle("UR5_joint6")
    
    print("Simple Pick and Place Script Started")
    print("Robot handles obtained")
    
    -- Start after 2 seconds
    sim.wait(2)
    do_pick_and_place()
end

function do_pick_and_place()
    print("Starting pick and place operation...")
    
    -- Step 1: Move to above Master Chef Can
    print("Step 1: Moving above Master Chef Can")
    sim.setJointTargetPosition(joint1, -0.5)
    sim.setJointTargetPosition(joint2, -0.8)
    sim.setJointTargetPosition(joint3, -1.2)
    sim.setJointTargetPosition(joint4, -1.5)
    sim.setJointTargetPosition(joint5, -1.6)
    sim.setJointTargetPosition(joint6, 0)
    sim.wait(3)
    
    -- Step 2: Open gripper
    print("Step 2: Opening gripper")
    sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', 1)
    sim.wait(1)
    
    -- Step 3: Move down to pick
    print("Step 3: Moving down to pick")
    sim.setJointTargetPosition(joint1, -0.5)
    sim.setJointTargetPosition(joint2, -0.9)
    sim.setJointTargetPosition(joint3, -1.3)
    sim.setJointTargetPosition(joint4, -1.5)
    sim.setJointTargetPosition(joint5, -1.6)
    sim.setJointTargetPosition(joint6, 0)
    sim.wait(3)
    
    -- Step 4: Close gripper
    print("Step 4: Closing gripper")
    sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', 0)
    sim.wait(1)
    
    -- Step 5: Lift up
    print("Step 5: Lifting up")
    sim.setJointTargetPosition(joint1, -0.5)
    sim.setJointTargetPosition(joint2, -0.8)
    sim.setJointTargetPosition(joint3, -1.2)
    sim.setJointTargetPosition(joint4, -1.5)
    sim.setJointTargetPosition(joint5, -1.6)
    sim.setJointTargetPosition(joint6, 0)
    sim.wait(3)
    
    -- Step 6: Move to place position
    print("Step 6: Moving to place position")
    sim.setJointTargetPosition(joint1, 0.5)
    sim.setJointTargetPosition(joint2, -0.8)
    sim.setJointTargetPosition(joint3, -1.2)
    sim.setJointTargetPosition(joint4, -1.5)
    sim.setJointTargetPosition(joint5, -1.6)
    sim.setJointTargetPosition(joint6, 0)
    sim.wait(3)
    
    -- Step 7: Open gripper to place
    print("Step 7: Opening gripper to place")
    sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', 1)
    sim.wait(1)
    
    -- Step 8: Move back up
    print("Step 8: Moving back up")
    sim.setJointTargetPosition(joint1, 0.5)
    sim.setJointTargetPosition(joint2, -0.8)
    sim.setJointTargetPosition(joint3, -1.2)
    sim.setJointTargetPosition(joint4, -1.5)
    sim.setJointTargetPosition(joint5, -1.6)
    sim.setJointTargetPosition(joint6, 0)
    sim.wait(3)
    
    -- Step 9: Return to home
    print("Step 9: Returning to home")
    sim.setJointTargetPosition(joint1, 0)
    sim.setJointTargetPosition(joint2, -1.57)
    sim.setJointTargetPosition(joint3, -1.57)
    sim.setJointTargetPosition(joint4, -1.57)
    sim.setJointTargetPosition(joint5, -1.57)
    sim.setJointTargetPosition(joint6, 0)
    sim.wait(3)
    
    print("Pick and place operation completed!")
end

function sysCall_actuation()
    -- Do nothing here
end

function sysCall_cleanup()
    print("Script cleanup")
end




