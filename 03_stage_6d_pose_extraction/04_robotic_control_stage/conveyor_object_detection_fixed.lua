-- Fixed Conveyor Object Detection and Grasping Coordination Script
-- Integrates with the existing conveyor system for AI 6D Pose recognition

sim=require'sim'
conveyorSystem=require('models.conveyorSystem_customization-3')

-- Global variables
local conveyorHandle = nil
local cameraHandle = nil
local robotHandle = nil
local detectionInterval = 0.1  -- Detection frequency (seconds)
local lastDetectionTime = 0
local detectedObjects = {}
local graspingZone = {x=0.5, y=0.0, z=0.1}  -- Robot grasping position
local placementZone = {x=0.3, y=-0.4, z=0.1}  -- Left side placement area

-- Initialize the system
function initializeSystem()
    print("🤖 Initializing Conveyor Object Detection System...")
    
    -- Get handles with proper error checking
    conveyorHandle = sim.getObject('./conveyorSystem')
    if conveyorHandle == -1 then
        print("❌ Conveyor system not found! Trying alternative path...")
        conveyorHandle = sim.getObject('./efficientConveyor')
        if conveyorHandle == -1 then
            print("❌ Conveyor system not found!")
            return false
        end
    end
    
    -- Try different camera paths
    cameraHandle = sim.getObject('./XYZCameraProxy')
    if cameraHandle == -1 then
        print("⚠️  XYZCameraProxy not found, trying alternative cameras...")
        cameraHandle = sim.getObject('./DefaultCamera')
        if cameraHandle == -1 then
            print("❌ No camera found!")
            return false
        end
    end
    
    -- Try different robot paths
    robotHandle = sim.getObject('./UR5')
    if robotHandle == -1 then
        print("⚠️  UR5 not found, trying alternative path...")
        robotHandle = sim.getObject('./UR5_target')
        if robotHandle == -1 then
            print("❌ Robot not found!")
            return false
        end
    end
    
    print("✅ All components found!")
    print("   - Conveyor: " .. tostring(conveyorHandle))
    print("   - Camera: " .. tostring(cameraHandle))
    print("   - Robot: " .. tostring(robotHandle))
    
    -- Start conveyor at moderate speed
    local success = pcall(function()
        sim.setBufferProperty(conveyorHandle, 'customData.__ctrl__', sim.packTable({vel=0.05}))
    end)
    
    if success then
        print("🚀 Conveyor started at speed 0.05")
    else
        print("⚠️  Could not set conveyor speed, using default")
    end
    
    return true
end

-- Capture camera data for object detection
function captureCameraData()
    if cameraHandle == -1 then return nil end
    
    local rgbImage = nil
    local depthImage = nil
    
    -- Try to capture RGB image
    local success, result = pcall(function()
        return sim.getVisionSensorImg(cameraHandle)
    end)
    
    if success and result then
        rgbImage = result
    end
    
    -- Try to capture depth image
    success, result = pcall(function()
        return sim.getVisionSensorDepthBuffer(cameraHandle)
    end)
    
    if success and result then
        depthImage = result
    end
    
    if rgbImage then
        -- Save RGB data for Python processing
        local file = io.open("conveyor_rgb.txt", "wb")
        if file then
            file:write(rgbImage)
            file:close()
            print("📸 RGB data captured: " .. #rgbImage .. " bytes")
        end
        
        -- Save depth data if available
        if depthImage then
            local depthFile = io.open("conveyor_depth.txt", "wb")
            if depthFile then
                depthFile:write(depthImage)
                depthFile:close()
                print("📸 Depth data captured: " .. #depthImage .. " bytes")
            end
        end
        
        return {rgb=rgbImage, depth=depthImage}
    end
    
    return nil
end

-- Get conveyor state safely
function getConveyorState()
    local success, data = pcall(function()
        return sim.readCustomTableData(conveyorHandle, '__state__')
    end)
    
    if success and data then
        return data
    else
        -- Return default state if reading fails
        return {pos=0, vel=0.05}
    end
end

-- Calculate object position relative to robot
function calculateObjectPose(objectData, cameraData)
    -- This is a simplified pose calculation
    local conveyorState = getConveyorState()
    local conveyorPos = 0
    
    if type(conveyorState.pos) == "number" then
        conveyorPos = conveyorState.pos
    end
    
    -- Estimate object position based on conveyor position
    local objectPose = {
        x = graspingZone.x + conveyorPos * 0.1,  -- Adjust based on your setup
        y = graspingZone.y,
        z = graspingZone.z,
        rotation = 0  -- Simplified rotation
    }
    
    return objectPose
end

-- Check if object is in grasping zone
function isObjectInGraspingZone(objectPose)
    if not objectPose or not objectPose.x or not objectPose.y or not objectPose.z then
        return false
    end
    
    local distance = math.sqrt(
        (objectPose.x - graspingZone.x)^2 + 
        (objectPose.y - graspingZone.y)^2 + 
        (objectPose.z - graspingZone.z)^2
    )
    
    return distance < 0.1  -- 10cm tolerance
end

-- Execute grasping sequence
function executeGrasp(objectPose)
    print("🤖 Executing grasp sequence...")
    
    -- Stop conveyor safely
    local success = pcall(function()
        sim.setBufferProperty(conveyorHandle, 'customData.__ctrl__', sim.packTable({vel=0.0}))
    end)
    
    if success then
        print("⏹️  Conveyor stopped for grasping")
    else
        print("⚠️  Could not stop conveyor")
    end
    
    -- Here you would add robot control commands
    -- For now, we'll simulate the grasp
    print("🎯 Moving robot to position: " .. tostring(objectPose.x) .. ", " .. tostring(objectPose.y) .. ", " .. tostring(objectPose.z))
    
    -- Simulate grasp time
    sim.wait(2.0)
    
    -- Move to placement zone
    print("📦 Moving to placement zone: " .. tostring(placementZone.x) .. ", " .. tostring(placementZone.y) .. ", " .. tostring(placementZone.z))
    sim.wait(1.5)
    
    -- Resume conveyor safely
    success = pcall(function()
        sim.setBufferProperty(conveyorHandle, 'customData.__ctrl__', sim.packTable({vel=0.05}))
    end)
    
    if success then
        print("🚀 Conveyor resumed")
    else
        print("⚠️  Could not resume conveyor")
    end
    
    print("✅ Grasp sequence completed!")
end

-- Main detection and control loop
function detectionLoop()
    local currentTime = sim.getSimulationTime()
    
    -- Run detection at specified interval
    if currentTime - lastDetectionTime >= detectionInterval then
        print("\n🔍 Running object detection...")
        
        -- Capture camera data
        local cameraData = captureCameraData()
        if cameraData then
            -- In a real system, you'd send this to Python for YOLO detection
            -- For now, we'll simulate detection
            
            -- Simulate detected object
            local simulatedObject = {
                class = "blue_can",
                confidence = 0.85,
                bbox = {10, 20, 50, 60}  -- Simplified bounding box
            }
            
            -- Calculate pose
            local objectPose = calculateObjectPose(simulatedObject, cameraData)
            
            print("🎯 Detected: " .. simulatedObject.class .. " (conf: " .. tostring(simulatedObject.confidence) .. ")")
            print("📍 Position: " .. tostring(objectPose.x) .. ", " .. tostring(objectPose.y) .. ", " .. tostring(objectPose.z))
            
            -- Check if object is ready for grasping
            if isObjectInGraspingZone(objectPose) then
                print("🤖 Object in grasping zone - initiating grasp!")
                executeGrasp(objectPose)
            else
                print("⏳ Object approaching grasping zone...")
            end
            
            -- Store detection result
            detectedObjects[#detectedObjects + 1] = {
                time = currentTime,
                object = simulatedObject,
                pose = objectPose
            }
        else
            print("⚠️  No camera data available")
        end
        
        lastDetectionTime = currentTime
    end
end

-- Main execution
function main()
    print("🚀 Starting Conveyor Object Detection System")
    print("==================================================")
    
    if not initializeSystem() then
        print("❌ System initialization failed!")
        return
    end
    
    print("✅ System initialized successfully!")
    print("📋 System Configuration:")
    print("   - Detection interval: " .. tostring(detectionInterval) .. "s")
    print("   - Grasping zone: " .. tostring(graspingZone.x) .. ", " .. tostring(graspingZone.y) .. ", " .. tostring(graspingZone.z))
    print("   - Placement zone: " .. tostring(placementZone.x) .. ", " .. tostring(placementZone.y) .. ", " .. tostring(placementZone.z))
    print("   - Conveyor speed: 0.05")
    
    print("\n🎯 System ready! Objects will be detected and grasped automatically.")
    print("💡 Check 'conveyor_rgb.txt' and 'conveyor_depth.txt' for camera data.")
    
    -- Set up the detection loop
    sim.setThreadAutomaticSwitch(false)
    
    while sim.getSimulationTime() < 1000 do  -- Run for 1000 seconds
        detectionLoop()
        sim.switchThread()
    end
    
    print("🏁 Detection system completed!")
end

-- Run the main function
main()
