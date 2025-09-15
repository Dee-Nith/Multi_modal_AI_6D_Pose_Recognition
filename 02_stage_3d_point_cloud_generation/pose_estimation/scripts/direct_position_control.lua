-- Direct Position Control Script
-- Moves robot directly to coordinates without complex joint calculations

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
    
    print("Direct Position Control Script Started")
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
    
    print("Ready to start direct position control operation")
end

function sysCall_actuation()
    -- This runs every simulation step
    step_timer = step_timer + sim.getSimulationTimeStep()
    
    -- Start operation after 2 seconds
    if current_step == 0 and step_timer > 2 then
        current_step = 1
        print("Starting direct position control operation...")
    end
    
    -- Execute steps based on timer
    if current_step == 1 and step_timer > 5 then
        -- Step 1: Move to above Master Chef Can (10cm above)
        print("Step 1: Moving above Master Chef Can")
        move_robot_to_position_direct(master_chef_x, master_chef_y, master_chef_z + 0.10)
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
        move_robot_to_position_direct(master_chef_x, master_chef_y, master_chef_z)
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
        move_robot_to_position_direct(master_chef_x, master_chef_y, master_chef_z + 0.15)
        current_step = 6
        print("Step 5 completed")
    elseif current_step == 6 and step_timer > 19 then
        -- Step 6: Move to place conveyor (10cm above)
        print("Step 6: Moving to place conveyor")
        move_robot_to_position_direct(place_x, place_y, place_z + 0.10)
        current_step = 7
        print("Step 6 completed")
    elseif current_step == 7 and step_timer > 22 then
        -- Step 7: Move down to place position
        print("Step 7: Moving down to place position")
        move_robot_to_position_direct(place_x, place_y, place_z)
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
        move_robot_to_position_direct(robot_base_x, robot_base_y, robot_base_z + 0.30)
        current_step = 10
        print("Step 9 completed")
    elseif current_step == 10 and step_timer > 31 then
        print("Direct position control operation completed!")
        print("Master Chef Can picked from: (" .. master_chef_x .. ", " .. master_chef_y .. ", " .. master_chef_z .. ")")
        print("Master Chef Can placed at: (" .. place_x .. ", " .. place_y .. ", " .. place_z .. ")")
        current_step = 11
    end
end

function move_robot_to_position_direct(x, y, z)
    -- Move robot directly to world coordinates
    -- This uses sim.setObjectPosition to move the robot base
    
    print("🤖 Moving robot to position: (" .. x .. ", " .. y .. ", " .. z .. ")")
    
    -- Get current robot position
    local current_pos = sim.getObjectPosition(robot, -1)
    print("Current robot position: (" .. current_pos[1] .. ", " .. current_pos[2] .. ", " .. current_pos[3] .. ")")
    
    -- Set target position
    local target_pos = {x, y, z}
    sim.setObjectPosition(robot, -1, target_pos)
    
    print("✅ Robot position set to: (" .. x .. ", " .. y .. ", " .. z .. ")")
    
    -- Wait a bit for movement to complete
    sim.wait(0.5)
end

function sysCall_cleanup()
    print("Script cleanup")
end
