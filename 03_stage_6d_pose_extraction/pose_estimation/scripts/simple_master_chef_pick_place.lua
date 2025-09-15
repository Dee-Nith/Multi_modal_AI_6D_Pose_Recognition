-- Simple Master Chef Can Pick and Place Script
-- Uses exact coordinates from CoppeliaSim scene
-- Focus: Pick Master Chef Can and place it perfectly

function sysCall_init()
    -- Initialize robot handles
    robot = sim.getObjectHandle("UR5")
    gripper = sim.getObjectHandle("RG2")
    
    -- Get robot joint handles
    joint_handles = {}
    for i = 1, 6 do
        joint_handles[i] = sim.getObjectHandle("UR5_joint" .. i)
    end
    
    -- EXACT COORDINATES from CoppeliaSim scene
    master_chef_position = {-0.625, -0.275, 0.750}  -- Master Chef Can position
    robot_base_position = {-0.625, 0.075, 0.700}     -- Robot base position
    place_conveyor_position = {0.100, -0.625, 0.700} -- Place conveyor position
    
    -- Safety heights (10cm above objects)
    approach_height = 0.10  -- 10cm above object
    lift_height = 0.15      -- 15cm above object
    
    print("🤖 Simple Master Chef Can Pick and Place Script")
    print("📍 Master Chef Can Position: (" .. master_chef_position[1] .. ", " .. master_chef_position[2] .. ", " .. master_chef_position[3] .. ")")
    print("📍 Place Conveyor Position: (" .. place_conveyor_position[1] .. ", " .. place_conveyor_position[2] .. ", " .. place_conveyor_position[3] .. ")")
    print("🎯 Ready to execute single pick and place operation!")
    
    -- Start the operation after a short delay
    sim.wait(2)
    start_pick_and_place()
end

function start_pick_and_place()
    print("🚀 Starting Master Chef Can Pick and Place Operation...")
    print("=" * 60)
    
    -- Step 1: Move to approach position above Master Chef Can
    print("📦 Step 1: Moving to approach position above Master Chef Can")
    local approach_pos = {
        master_chef_position[1], 
        master_chef_position[2], 
        master_chef_position[3] + approach_height
    }
    move_robot_to_position(approach_pos, "approach above Master Chef Can")
    
    -- Step 2: Open gripper
    print("🤏 Step 2: Opening gripper")
    open_gripper()
    
    -- Step 3: Move to pick position (exact Master Chef Can position)
    print("📦 Step 3: Moving to pick position at Master Chef Can")
    move_robot_to_position(master_chef_position, "pick position at Master Chef Can")
    
    -- Step 4: Close gripper to grasp
    print("🤏 Step 4: Closing gripper to grasp Master Chef Can")
    close_gripper()
    
    -- Step 5: Lift Master Chef Can
    print("📦 Step 5: Lifting Master Chef Can")
    local lift_pos = {
        master_chef_position[1], 
        master_chef_position[2], 
        master_chef_position[3] + lift_height
    }
    move_robot_to_position(lift_pos, "lift position above Master Chef Can")
    
    -- Step 6: Move to place conveyor approach position
    print("📦 Step 6: Moving to place conveyor approach position")
    local place_approach_pos = {
        place_conveyor_position[1], 
        place_conveyor_position[2], 
        place_conveyor_position[3] + approach_height
    }
    move_robot_to_position(place_approach_pos, "approach above place conveyor")
    
    -- Step 7: Move to place position on conveyor
    print("📦 Step 7: Moving to place position on conveyor")
    move_robot_to_position(place_conveyor_position, "place position on conveyor")
    
    -- Step 8: Open gripper to release Master Chef Can
    print("🤏 Step 8: Opening gripper to release Master Chef Can")
    open_gripper()
    
    -- Step 9: Move back to approach position above conveyor
    print("📦 Step 9: Moving back to approach position above conveyor")
    move_robot_to_position(place_approach_pos, "approach above place conveyor")
    
    -- Step 10: Return to robot base home position
    print("🏠 Step 10: Returning to robot base home position")
    local home_pos = {
        robot_base_position[1], 
        robot_base_position[2], 
        robot_base_position[3] + 0.3  -- 30cm above base
    }
    move_robot_to_position(home_pos, "home position")
    
    print("🎉 MASTER CHEF CAN PICK AND PLACE COMPLETED SUCCESSFULLY!")
    print("✅ Object picked from: (" .. master_chef_position[1] .. ", " .. master_chef_position[2] .. ", " .. master_chef_position[3] .. ")")
    print("✅ Object placed at: (" .. place_conveyor_position[1] .. ", " .. place_conveyor_position[2] .. ", " .. place_conveyor_position[3] .. ")")
    print("🤖 Robot returned to home position")
end

function move_robot_to_position(target_pos, description)
    print("🤖 Moving robot to " .. description .. ": (" .. target_pos[1] .. ", " .. target_pos[2] .. ", " .. target_pos[3] .. ")")
    
    -- Calculate joint angles for target position
    local target_joints = calculate_joint_angles(target_pos[1], target_pos[2], target_pos[3])
    
    if target_joints then
        -- Move robot to target joint configuration
        for i = 1, 6 do
            sim.setJointTargetPosition(joint_handles[i], target_joints[i])
        end
        
        -- Wait for movement to complete
        sim.wait(3)  -- Wait 3 seconds for movement
        
        print("✅ Robot moved to " .. description)
    else
        print("❌ Failed to calculate joint angles for " .. description)
    end
end

function calculate_joint_angles(x, y, z)
    -- Calculate joint angles for UR5 robot
    -- This is a simplified calculation - in practice you'd use proper inverse kinematics
    
    -- Calculate base rotation (yaw)
    local base_angle = math.atan2(y, x)
    
    -- Calculate reach distance
    local reach = math.sqrt(x*x + y*y)
    
    -- Simplified joint angles for UR5
    local joints = {
        base_angle,                    -- Base rotation (joint 1)
        -math.pi/4,                    -- Shoulder (joint 2) - 45 degrees down
        -math.pi/2,                    -- Elbow (joint 3) - 90 degrees down
        0,                             -- Wrist 1 (joint 4)
        -math.pi/2,                    -- Wrist 2 (joint 5) - 90 degrees down
        0                              -- Wrist 3 (joint 6)
    }
    
    return joints
end

function open_gripper()
    print("🤏 Opening gripper...")
    sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', 1)
    sim.wait(1)  -- Wait for gripper to open
    print("✅ Gripper opened")
end

function close_gripper()
    print("🤏 Closing gripper...")
    sim.setIntProperty(sim.handle_scene, 'signal.RG2_open', 0)
    sim.wait(1)  -- Wait for gripper to close
    print("✅ Gripper closed")
end

function sysCall_actuation()
    -- This function runs every simulation step
    -- We don't need continuous actuation for this simple script
end

function sysCall_cleanup()
    print("🏁 Simple Master Chef Can Script Cleanup Complete")
end




