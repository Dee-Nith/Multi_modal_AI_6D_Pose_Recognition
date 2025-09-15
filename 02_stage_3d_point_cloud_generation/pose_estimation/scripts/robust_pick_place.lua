-- Robust Pick and Place Script
-- More stable execution

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
    
    print("Robust Pick and Place Script Started")
    print("Robot handles obtained")
    
    -- Initialize step counter
    current_step = 0
    step_timer = 0
    
    print("Ready to start pick and place operation")
end

function sysCall_actuation()
    -- This runs every simulation step
    step_timer = step_timer + sim.getSimulationTimeStep()
    
    -- Start operation after 2 seconds
    if current_step == 0 and step_timer > 2 then
        current_step = 1
        print("Starting pick and place operation...")
    end
    
    -- Execute steps based on timer
    if current_step == 1 and step_timer > 5 then
        -- Step 1: Move to above Master Chef Can
        print("Step 1: Moving above Master Chef Can")
        sim.setJointTargetPosition(joint1, -0.5)
        sim.setJointTargetPosition(joint2, -0.8)
        sim.setJointTargetPosition(joint3, -1.2)
        sim.setJointTargetPosition(joint4, -1.5)
        sim.setJointTargetPosition(joint5, -1.6)
        sim.setJointTargetPosition(joint6, 0)
        current_step = 2
        print("Step 1 completed")
    elseif current_step == 2 and step_timer > 8 then
        -- Step 2: Open gripper
        print("Step 2: Opening gripper")
        sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', 1)
        current_step = 3
        print("Step 2 completed")
    elseif current_step == 3 and step_timer > 10 then
        -- Step 3: Move down to pick
        print("Step 3: Moving down to pick")
        sim.setJointTargetPosition(joint1, -0.5)
        sim.setJointTargetPosition(joint2, -0.9)
        sim.setJointTargetPosition(joint3, -1.3)
        sim.setJointTargetPosition(joint4, -1.5)
        sim.setJointTargetPosition(joint5, -1.6)
        sim.setJointTargetPosition(joint6, 0)
        current_step = 4
        print("Step 3 completed")
    elseif current_step == 4 and step_timer > 13 then
        -- Step 4: Close gripper
        print("Step 4: Closing gripper")
        sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', 0)
        current_step = 5
        print("Step 4 completed")
    elseif current_step == 5 and step_timer > 16 then
        -- Step 5: Lift up
        print("Step 5: Lifting up")
        sim.setJointTargetPosition(joint1, -0.5)
        sim.setJointTargetPosition(joint2, -0.8)
        sim.setJointTargetPosition(joint3, -1.2)
        sim.setJointTargetPosition(joint4, -1.5)
        sim.setJointTargetPosition(joint5, -1.6)
        sim.setJointTargetPosition(joint6, 0)
        current_step = 6
        print("Step 5 completed")
    elseif current_step == 6 and step_timer > 19 then
        -- Step 6: Move to place position
        print("Step 6: Moving to place position")
        sim.setJointTargetPosition(joint1, 0.5)
        sim.setJointTargetPosition(joint2, -0.8)
        sim.setJointTargetPosition(joint3, -1.2)
        sim.setJointTargetPosition(joint4, -1.5)
        sim.setJointTargetPosition(joint5, -1.6)
        sim.setJointTargetPosition(joint6, 0)
        current_step = 7
        print("Step 6 completed")
    elseif current_step == 7 and step_timer > 22 then
        -- Step 7: Open gripper to place
        print("Step 7: Opening gripper to place")
        sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', 1)
        current_step = 8
        print("Step 7 completed")
    elseif current_step == 8 and step_timer > 25 then
        -- Step 8: Return to home
        print("Step 8: Returning to home")
        sim.setJointTargetPosition(joint1, 0)
        sim.setJointTargetPosition(joint2, -1.57)
        sim.setJointTargetPosition(joint3, -1.57)
        sim.setJointTargetPosition(joint4, -1.57)
        sim.setJointTargetPosition(joint5, -1.57)
        sim.setJointTargetPosition(joint6, 0)
        current_step = 9
        print("Step 8 completed")
    elseif current_step == 9 and step_timer > 28 then
        print("Pick and place operation completed!")
        current_step = 10
    end
end

function sysCall_cleanup()
    print("Script cleanup")
end




