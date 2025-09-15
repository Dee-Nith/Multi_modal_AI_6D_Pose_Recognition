-- Conveyor Object Detection and Grasping Coordination Script
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
    
    -- Get handles
    conveyorHandle = sim.getObject('./conveyorSystem')
    cameraHandle = sim.getObject('./XYZCameraProxy')  -- Adjust path as needed
    robotHandle = sim.getObject('./UR5')  -- Adjust path as needed
    
    if conveyorHandle == -1 then
        print("❌ Conveyor system not found!")
        return false
    end
    
    if cameraHandle == -1 then
        print("❌ Camera not found!")
        return false
    end
    
    if robotHandle == -1 then
        print("❌ Robot not found!")
        return false
    end
    
    print("✅ All components found!")
    
    -- Start conveyor at moderate speed
    sim.setBufferProperty(conveyorHandle, 'customData.__ctrl__', sim.packTable({vel=0.05}))
    print("🚀 Conveyor started at speed 0.05")
    
    return true
end

-- Capture camera data for object detection
function captureCameraData()
    if cameraHandle == -1 then return nil end
    
    local rgbImage = sim.getVisionSensorImg(cameraHandle)
    local depthImage = sim.getVisionSensorDepthBuffer(cameraHandle)
    
    if rgbImage and depthImage then
        -- Save RGB data for Python processing
        local file = io.open("conveyor_rgb.txt", "wb")
        if file then
            file:write(rgbImage)
            file:close()
        end
        
        -- Save depth data
        local depthFile = io.open("conveyor_depth.txt", "wb")
        if depthFile then
            depthFile:write(depthImage)
            depthFile:close()
        end
        
        print("📸 Camera data captured: " .. #rgbImage .. " RGB, " .. #depthImage .. " depth")
        return {rgb=rgbImage, depth=depthImage}
    end
    
    return nil
end

-- Get conveyor state
function getConveyorState()
    local data = sim.readCustomTableData(conveyorHandle, '__state__')
    return data
end

-- Calculate object position relative to robot
function calculateObjectPose(objectData, cameraData)
    -- This is a simplified pose calculation
    -- In a real system, you'd use the depth data for 3D positioning
    
    local conveyorState = getConveyorState()
    local conveyorPos = conveyorState.pos or 0
    
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
    
    -- Stop conveyor
    sim.setBufferProperty(conveyorHandle, 'customData.__ctrl__', sim.packTable({vel=0.0}))
    print("⏹️  Conveyor stopped for grasping")
    
    -- Here you would add robot control commands
    -- For now, we'll simulate the grasp
    print("🎯 Moving robot to position: " .. objectPose.x .. ", " .. objectPose.y .. ", " .. objectPose.z)
    
    -- Simulate grasp time
    sim.wait(2.0)
    
    -- Move to placement zone
    print("📦 Moving to placement zone: " .. placementZone.x .. ", " .. placementZone.y .. ", " .. placementZone.z)
    sim.wait(1.5)
    
    -- Resume conveyor
    sim.setBufferProperty(conveyorHandle, 'customData.__ctrl__', sim.packTable({vel=0.05}))
    print("🚀 Conveyor resumed")
    
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
            
            print("🎯 Detected: " .. simulatedObject.class .. " (conf: " .. simulatedObject.confidence .. ")")
            print("📍 Position: " .. objectPose.x .. ", " .. objectPose.y .. ", " .. objectPose.z)
            
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
        end
        
        lastDetectionTime = currentTime
    end
end

-- Main execution
function main()
    print("🚀 Starting Conveyor Object Detection System")
    print("=" * 50)
    
    if not initializeSystem() then
        print("❌ System initialization failed!")
        return
    end
    
    print("✅ System initialized successfully!")
    print("📋 System Configuration:")
    print("   - Detection interval: " .. detectionInterval .. "s")
    print("   - Grasping zone: " .. graspingZone.x .. ", " .. graspingZone.y .. ", " .. graspingZone.z)
    print("   - Placement zone: " .. placementZone.x .. ", " .. placementZone.y .. ", " .. placementZone.z)
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




