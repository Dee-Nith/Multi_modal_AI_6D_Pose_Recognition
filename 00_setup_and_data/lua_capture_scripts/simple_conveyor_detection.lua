-- Simple Conveyor Object Detection Script
-- Basic version that works with the current scene

sim=require'sim'

print("🚀 Starting Simple Conveyor Detection System")
print("=============================================")

-- Simple initialization without complex path finding
function initializeSimple()
    print("🤖 Initializing simple detection system...")
    
    -- Try to find objects with simple paths
    local conveyorHandle = sim.getObject('.')
    print("✅ Using current object as conveyor handle: " .. tostring(conveyorHandle))
    
    -- Try to find camera
    local cameraHandle = sim.getObject('./DefaultCamera')
    if cameraHandle == -1 then
        print("⚠️  DefaultCamera not found, trying XYZCameraProxy...")
        cameraHandle = sim.getObject('./XYZCameraProxy')
        if cameraHandle == -1 then
            print("⚠️  No camera found, will simulate detection")
            cameraHandle = nil
        else
            print("✅ Found camera: " .. tostring(cameraHandle))
        end
    else
        print("✅ Found DefaultCamera: " .. tostring(cameraHandle))
    end
    
    return conveyorHandle, cameraHandle
end

-- Simple camera capture
function captureSimpleCamera(cameraHandle)
    if not cameraHandle then
        print("📸 Simulating camera capture (no camera available)")
        return true
    end
    
    local success, rgbImage = pcall(function()
        return sim.getVisionSensorImg(cameraHandle)
    end)
    
    if success and rgbImage then
        -- Save RGB data
        local file = io.open("simple_conveyor_rgb.txt", "wb")
        if file then
            file:write(rgbImage)
            file:close()
            print("📸 Camera data saved: " .. #rgbImage .. " bytes")
            return true
        end
    else
        print("⚠️  Camera capture failed, simulating...")
        return true
    end
    
    return false
end

-- Simple object detection simulation
function simulateDetection()
    print("🔍 Simulating object detection...")
    
    -- Simulate detected object
    local detectedObject = {
        class = "blue_can",
        confidence = 0.85,
        position = {x = 0.5, y = 0.0, z = 0.1}
    }
    
    print("🎯 Detected: " .. detectedObject.class)
    print("   Confidence: " .. tostring(detectedObject.confidence))
    print("   Position: " .. tostring(detectedObject.position.x) .. ", " .. 
          tostring(detectedObject.position.y) .. ", " .. tostring(detectedObject.position.z))
    
    return detectedObject
end

-- Simple grasping simulation
function simulateGrasp(object)
    print("🤖 Simulating grasp sequence...")
    print("🎯 Moving to object: " .. object.class)
    
    -- Simulate grasp time
    sim.wait(1.0)
    
    print("📦 Moving to placement area...")
    sim.wait(1.0)
    
    print("✅ Grasp simulation completed!")
end

-- Main detection loop
function detectionLoop()
    print("\n🔄 Running detection cycle...")
    
    -- Capture camera data
    local captureSuccess = captureSimpleCamera(cameraHandle)
    
    if captureSuccess then
        -- Simulate detection
        local detectedObject = simulateDetection()
        
        -- Simulate grasping if object is detected
        if detectedObject.confidence > 0.5 then
            simulateGrasp(detectedObject)
        end
    end
    
    print("⏳ Waiting for next detection cycle...")
end

-- Main execution
print("🎯 Starting simple conveyor detection...")

-- Initialize
local conveyorHandle, cameraHandle = initializeSimple()

if conveyorHandle then
    print("✅ System initialized successfully!")
    print("📋 Configuration:")
    print("   - Conveyor: " .. tostring(conveyorHandle))
    print("   - Camera: " .. tostring(cameraHandle or "None (simulating)"))
    print("   - Detection: Simulated")
    
    print("\n🎯 System ready! Running detection simulation...")
    
    -- Run a few detection cycles
    for i = 1, 5 do
        print("\n--- Detection Cycle " .. i .. " ---")
        detectionLoop()
        sim.wait(2.0)  -- Wait 2 seconds between cycles
    end
    
    print("\n🏁 Simple detection system completed!")
    print("💡 Check 'simple_conveyor_rgb.txt' for camera data (if available)")
else
    print("❌ Failed to initialize system")
end

print("🤖 Simple conveyor detection script finished!")




