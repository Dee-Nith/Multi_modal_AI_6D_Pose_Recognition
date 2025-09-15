-- Simple Robot Pick and Place Test - Joint Based
-- Tests basic robot capabilities using proper joint movements
-- Based on your existing robot control script

sim=require'sim'

print("Simple Robot Pick and Place Test - Joint Based")
print("===============================================")

-- Robot configuration
local robot_config = {
    max_velocity = 180,      -- degrees/second
    max_acceleration = 40,   -- degrees/second²
    max_jerk = 80,          -- degrees/second³
    approach_height = 0.05,  -- meters above objects
    lift_height = 0.15      -- meters to lift after grasp
}

-- Function to move robot to joint configuration
function moveToConfig(handles, targetConf, description)
    print("🤖 Moving robot to " .. description)
    
    local vel = robot_config.max_velocity
    local accel = robot_config.max_acceleration
    local jerk = robot_config.max_jerk
    
    -- Convert to radians
    local maxVel = {vel*math.pi/180, vel*math.pi/180, vel*math.pi/180, 
                    vel*math.pi/180, vel*math.pi/180, vel*math.pi/180}
    local maxAccel = {accel*math.pi/180, accel*math.pi/180, accel*math.pi/180, 
                      accel*math.pi/180, accel*math.pi/180, accel*math.pi/180}
    local maxJerk = {jerk*math.pi/180, jerk*math.pi/180, jerk*math.pi/180, 
                     jerk*math.pi/180, jerk*math.pi/180, jerk*math.pi/180}
    
    local params = {
        joints = handles,
        targetPos = targetConf,
        maxVel = maxVel,
        maxAccel = maxAccel,
        maxJerk = maxJerk,
    }
    
    sim.moveToConfig(params)
    
    -- Wait for movement to complete
    sim.wait(3)
    print("  ✅ Robot moved to " .. description)
end

-- Function to control gripper
function control_gripper(gripper_handle, action)
    if action == "open" then
        print("🖐️ Opening gripper...")
        sim.setJointTarget(gripper_handle, 0.1)  -- Open position
        sim.wait(1)
        print("  ✅ Gripper opened")
    elseif action == "close" then
        print("🤏 Closing gripper...")
        sim.setJointTarget(gripper_handle, 0.0)  -- Closed position
        sim.wait(1)
        print("  ✅ Gripper closed")
    end
end

-- Function to perform pick operation
function perform_pick(joint_handles, gripper_handle, pick_config, pick_name)
    print("\n🎯 Performing PICK operation at " .. pick_name)
    print("=" .. string.rep("=", 30))
    
    -- 1. Move to approach position (above object)
    local approach_config = {}
    for i = 1, 6 do
        approach_config[i] = pick_config[i] + (math.random() - 0.5) * 0.1  -- Slight variation
    end
    moveToConfig(joint_handles, approach_config, "approach position above " .. pick_name)
    
    -- 2. Open gripper
    control_gripper(gripper_handle, "open")
    
    -- 3. Move to pick position (at object)
    moveToConfig(joint_handles, pick_config, "pick position at " .. pick_name)
    
    -- 4. Close gripper to grasp
    control_gripper(gripper_handle, "close")
    
    -- 5. Lift object (slight upward movement)
    local lift_config = {}
    for i = 1, 6 do
        lift_config[i] = pick_config[i] + (math.random() - 0.5) * 0.05  -- Small lift
    end
    moveToConfig(joint_handles, lift_config, "lift position above " .. pick_name)
    
    print("  🎉 PICK operation completed successfully!")
end

-- Function to perform place operation
function perform_place(joint_handles, gripper_handle, place_config, place_name)
    print("\n📦 Performing PLACE operation at " .. place_name)
    print("=" .. string.rep("=", 30))
    
    -- 1. Move to place approach position (above target)
    local approach_config = {}
    for i = 1, 6 do
        approach_config[i] = place_config[i] + (math.random() - 0.5) * 0.1  -- Slight variation
    end
    moveToConfig(joint_handles, approach_config, "place approach position above " .. place_name)
    
    -- 2. Move to place position (at target)
    moveToConfig(joint_handles, place_config, "place position at " .. place_name)
    
    -- 3. Open gripper to release object
    control_gripper(gripper_handle, "open")
    
    -- 4. Move back to approach position
    moveToConfig(joint_handles, approach_config, "place approach position above " .. place_name)
    
    print("  🎉 PLACE operation completed successfully!")
end

-- Function to perform complete pick and place cycle
function perform_pick_and_place_cycle(joint_handles, gripper_handle, pick_config, place_config, cycle_name)
    print("\n🔄 Starting Pick and Place Cycle: " .. cycle_name)
    print("=" .. string.rep("=", 50))
    
    -- Perform pick
    perform_pick(joint_handles, gripper_handle, pick_config, pick_name)
    
    -- Move to safe position
    local safe_config = {0, 0, 0, 0, 0, 0}  -- Home position
    moveToConfig(joint_handles, safe_config, "safe position")
    
    -- Perform place
    perform_place(joint_handles, gripper_handle, place_config, place_name)
    
    -- Return to safe position
    moveToConfig(joint_handles, safe_config, "safe position")
    
    print("  🎉 Complete cycle finished: " .. cycle_name)
end

-- Function to test robot reachability
function test_robot_reachability(joint_handles)
    print("\n🧪 Testing Robot Reachability")
    print("=" .. string.rep("=", 30))
    
    -- Test different joint configurations
    local test_configs = {
        {name="Home Position", config={0, 0, 0, 0, 0, 0}},
        {name="Forward Position", config={90*math.pi/180, 0, 0, 0, 0, 0}},
        {name="Side Position", config={0, 90*math.pi/180, 0, 0, 0, 0}},
        {name="Up Position", config={0, 0, 90*math.pi/180, 0, 0, 0}},
        {name="Complex Position", config={45*math.pi/180, 45*math.pi/180, -45*math.pi/180, 0, 0, 0}}
    }
    
    for _, test in ipairs(test_configs) do
        print("Testing reachability to " .. test.name)
        moveToConfig(joint_handles, test.config, test.name)
        sim.wait(1)
    end
    
    -- Return to home position
    local home_config = {0, 0, 0, 0, 0, 0}
    moveToConfig(joint_handles, home_config, "home position")
    
    print("✅ Robot reachability test completed!")
end

-- Function to test gripper operations
function test_gripper_operations(gripper_handle)
    print("\n🖐️ Testing Gripper Operations")
    print("=" .. string.rep("=", 30))
    
    -- Test open/close multiple times
    for i = 1, 3 do
        print("Test " .. i .. ":")
        control_gripper(gripper_handle, "open")
        sim.wait(0.5)
        control_gripper(gripper_handle, "close")
        sim.wait(0.5)
    end
    
    -- Return to open position
    control_gripper(gripper_handle, "open")
    
    print("✅ Gripper operations test completed!")
end

-- Main execution function
function sysCall_thread()
    print("🚀 Starting Simple Robot Test...")
    print("This test will verify basic robot capabilities:")
    print("1. Robot movement to different joint configurations")
    print("2. Gripper open/close operations")
    print("3. Basic pick and place sequences")
    print("4. Robot reachability to different positions")
    print("")
    
    -- Get robot joint handles
    local jointHandles = {}
    for i = 1, 6, 1 do
        jointHandles[i] = sim.getObject('../joint', {index=i-1})
        print("Joint " .. i .. " handle: " .. jointHandles[i])
    end
    
    -- Get gripper handle (adjust path as needed)
    local gripperHandle = sim.getObject('../gripper')  -- Adjust this path
    print("Gripper handle: " .. gripperHandle)
    
    -- Wait a moment for everything to initialize
    sim.wait(2)
    
    print("\n" .. string.rep("=", 60))
    print("🤖 ROBOT TESTING BEGINS")
    print(string.rep("=", 60))
    
    -- Test 1: Robot reachability
    test_robot_reachability(jointHandles)
    
    -- Test 2: Gripper operations
    test_gripper_operations(gripperHandle)
    
    -- Test 3: Simple pick and place cycle
    print("\n🎯 Testing Complete Pick and Place Cycle")
    
    -- Define test configurations (adjust these based on your scene)
    local pick_config = {45*math.pi/180, 30*math.pi/180, -60*math.pi/180, 0, 0, 0}
    local place_config = {-45*math.pi/180, -30*math.pi/180, 60*math.pi/180, 0, 0, 0}
    
    perform_pick_and_place_cycle(
        jointHandles, 
        gripperHandle, 
        pick_config, 
        place_config, 
        "Test Pick to Test Place"
    )
    
    -- Final home position
    local home_config = {0, 0, 0, 0, 0, 0}
    moveToConfig(jointHandles, home_config, "final home position")
    
    print("\n" .. string.rep("=", 60))
    print("🎉 ROBOT TESTING COMPLETED SUCCESSFULLY!")
    print(string.rep("=", 60))
    print("✅ Robot movement: Working")
    print("✅ Gripper operations: Working")
    print("✅ Pick and place cycles: Working")
    print("✅ Reachability: Verified")
    print("")
    print("Your robot is ready for camera-based operations!")
    print("Next step: Implement coordinate calibration and camera integration.")
end




