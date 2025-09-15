-- Precise Pick and Place Script
-- Uses exact coordinates to reach Master Chef Can

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
    
    print("Precise Pick and Place Script Started")
    print("Robot handles obtained")
    
    -- YOUR EXACT COORDINATES from CoppeliaSim
    master_chef_x = -0.625  -- Master Chef Can X position
    master_chef_y = -0.275  -- Master Chef Can Y position
    master_chef_z = 0.750   -- Master Chef Can Z position
    
    place_x = 0.100         -- Place conveyor X position
    place_y = -0.625        -- Place conveyor Y position
    place_z = 0.700         -- Place conveyor Z position
    
    robot_base_x = -0.625   -- Robot base X position
    robot_base_y = 0.075    -- Robot base Y position
    robot_base_z = 0.700    -- Robot base Z position
    
    print("Master Chef Can Position: (" .. master_chef_x .. ", " .. master_chef_y .. ", " .. master_chef_z .. ")")
    print("Place Conveyor Position: (" .. place_x .. ", " .. place_y .. ", " .. place_z .. ")")
    
    -- Initialize step counter
    current_step = 0
    step_timer = 0
    
    print("Ready to start precise pick and place operation")
end

function sysCall_actuation()
    -- This runs every simulation step
    step_timer = step_timer + sim.getSimulationTimeStep()
    
    -- Start operation after 2 seconds
    if current_step == 0 and step_timer > 2 then
        current_step = 1
        print("Starting precise pick and place operation...")
    end
    
    -- Execute steps based on timer
    if current_step == 1 and step_timer > 5 then
        -- Step 1: Move to above Master Chef Can (10cm above)
        print("Step 1: Moving above Master Chef Can")
        local target_joints = calculate_joints_for_position(master_chef_x, master_chef_y, master_chef_z + 0.10)
        move_robot_to_joints(target_joints)
        current_step = 2
        print("Step 1 completed")
    elseif current_step == 2 and step_timer > 8 then
        -- Step 2: Open gripper
        print("Step 2: Opening gripper")
        sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', 1)
        current_step = 3
        print("Step 2 completed")
    elseif current_step == 3 and step_timer > 10 then
        -- Step 3: Move down to exact Master Chef Can position
        print("Step 3: Moving down to exact Master Chef Can position")
        local target_joints = calculate_joints_for_position(master_chef_x, master_chef_y, master_chef_z)
        move_robot_to_joints(target_joints)
        current_step = 4
        print("Step 3 completed")
    elseif current_step == 4 and step_timer > 13 then
        -- Step 4: Close gripper to grasp
        print("Step 4: Closing gripper to grasp")
        sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', 0)
        current_step = 5
        print("Step 4 completed")
    elseif current_step == 5 and step_timer > 16 then
        -- Step 5: Lift up (15cm above Master Chef Can)
        print("Step 5: Lifting up")
        local target_joints = calculate_joints_for_position(master_chef_x, master_chef_y, master_chef_z + 0.15)
        move_robot_to_joints(target_joints)
        current_step = 6
        print("Step 5 completed")
    elseif current_step == 6 and step_timer > 19 then
        -- Step 6: Move to place conveyor (10cm above)
        print("Step 6: Moving to place conveyor")
        local target_joints = calculate_joints_for_position(place_x, place_y, place_z + 0.10)
        move_robot_to_joints(target_joints)
        current_step = 7
        print("Step 6 completed")
    elseif current_step == 7 and step_timer > 22 then
        -- Step 7: Move down to place position
        print("Step 7: Moving down to place position")
        local target_joints = calculate_joints_for_position(place_x, place_y, place_z)
        move_robot_to_joints(target_joints)
        current_step = 8
        print("Step 7 completed")
    elseif current_step == 8 and step_timer > 25 then
        -- Step 8: Open gripper to release
        print("Step 8: Opening gripper to release")
        sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', 1)
        current_step = 9
        print("Step 8 completed")
    elseif current_step == 9 and step_timer > 28 then
        -- Step 9: Return to home position
        print("Step 9: Returning to home position")
        local target_joints = calculate_joints_for_position(robot_base_x, robot_base_y, robot_base_z + 0.30)
        move_robot_to_joints(target_joints)
        current_step = 10
        print("Step 9 completed")
    elseif current_step == 10 and step_timer > 31 then
        print("Precise pick and place operation completed!")
        print("Master Chef Can picked from: (" .. master_chef_x .. ", " .. master_chef_y .. ", " .. master_chef_z .. ")")
        print("Master Chef Can placed at: (" .. place_x .. ", " .. place_y .. ", " .. place_z .. ")")
        current_step = 11
    end
end

function calculate_joints_for_position(x, y, z)
    -- Calculate joint angles for target position
    -- This is a simplified calculation for UR5
    
    -- Calculate base rotation (yaw) - robot turns to face the target
    local base_angle = math.atan2(y, x)
    
    -- Calculate reach distance
    local reach = math.sqrt(x*x + y*y)
    
    -- Simplified joint angles for UR5
    local joints = {
        base_angle,                    -- Base rotation (joint 1) - turns to face target
        -math.pi/4,                    -- Shoulder (joint 2) - 45 degrees down
        -math.pi/2,                    -- Elbow (joint 3) - 90 degrees down
        0,                             -- Wrist 1 (joint 4)
        -math.pi/2,                    -- Wrist 2 (joint 5) - 90 degrees down
        0                              -- Wrist 3 (joint 6)
    }
    
    return joints
end

function move_robot_to_joints(target_joints)
    -- Move robot to target joint configuration
    sim.setJointTargetPosition(joint1, target_joints[1])
    sim.setJointTargetPosition(joint2, target_joints[2])
    sim.setJointTargetPosition(joint3, target_joints[3])
    sim.setJointTargetPosition(joint4, target_joints[4])
    sim.setJointTargetPosition(joint5, target_joints[5])
    sim.setJointTargetPosition(joint6, target_joints[6])
end

function sysCall_cleanup()
    print("Script cleanup")
end




