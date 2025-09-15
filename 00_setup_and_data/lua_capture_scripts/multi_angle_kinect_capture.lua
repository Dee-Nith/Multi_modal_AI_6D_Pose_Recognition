-- Multi-Angle Kinect RGB and Depth Capture Script
-- Captures data from 8 different angles around the objects
-- Author: AI Assistant
-- Date: 2024

function sysCall_init()
    print("🚀 Initializing Multi-Angle Kinect Capture...")
    
    -- Get camera handles
    kinectDepth = sim.getObject('./Kinect/DepthCamera')
    kinectRGB = sim.getObject('./Kinect/RGBCamera') 
    kinectRef = sim.getObject('./Kinect')
    
    if kinectDepth == -1 or kinectRGB == -1 or kinectRef == -1 then
        print("❌ Error: Kinect camera not found!")
        return
    end
    
    -- Store original position and orientation
    originalPosition = sim.getObjectPosition(kinectRef, -1)
    originalOrientation = sim.getObjectOrientation(kinectRef, -1)
    
    print("📍 Original Kinect position:", originalPosition[1], originalPosition[2], originalPosition[3])
    print("📐 Original Kinect orientation:", originalOrientation[1], originalOrientation[2], originalOrientation[3])
    
    -- Define capture angles (8 views around 360°)
    captureAngles = {0, 45, 90, 135, 180, 225, 270, 315}
    currentAngleIndex = 1
    captureCount = 0
    
    -- Capture parameters
    centerPoint = {-0.625, 0.075, 0.750}  -- Center of object area
    captureDistance = 1.0  -- Distance from center
    captureHeight = 0.8    -- Height above objects
    
    -- State management
    state = "MOVING"
    moveStartTime = sim.getSimulationTime()
    waitTime = 2.0  -- Wait time at each position
    
    print("✅ Multi-angle capture initialized!")
    print("📸 Will capture from", #captureAngles, "different angles")
    
    -- Start first capture
    moveToNextAngle()
end

function moveToNextAngle()
    if currentAngleIndex <= #captureAngles then
        local angle = captureAngles[currentAngleIndex]
        local angleRad = math.rad(angle)
        
        -- Calculate new position around the center point
        local newX = centerPoint[1] + captureDistance * math.cos(angleRad)
        local newY = centerPoint[2] + captureDistance * math.sin(angleRad)
        local newZ = centerPoint[3] + captureHeight
        
        -- Calculate orientation to look at center
        local dirX = centerPoint[1] - newX
        local dirY = centerPoint[2] - newY
        local dirZ = centerPoint[3] - newZ
        
        -- Calculate yaw to face center
        local yaw = math.atan2(dirY, dirX)
        -- Calculate pitch to look down at objects
        local horizontalDist = math.sqrt(dirX*dirX + dirY*dirY)
        local pitch = math.atan2(-dirZ, horizontalDist)
        
        -- Set new position and orientation
        sim.setObjectPosition(kinectRef, -1, {newX, newY, newZ})
        sim.setObjectOrientation(kinectRef, -1, {0, pitch, yaw})
        
        print(string.format("📸 Moving to angle %d° (%.2f, %.2f, %.2f)", 
              angle, newX, newY, newZ))
        
        state = "MOVING"
        moveStartTime = sim.getSimulationTime()
    else
        -- All captures complete, return to original position
        print("🎉 All captures complete! Returning to original position...")
        sim.setObjectPosition(kinectRef, -1, originalPosition)
        sim.setObjectOrientation(kinectRef, -1, originalOrientation)
        print("✅ Multi-angle capture session finished!")
    end
end

function captureCurrentView()
    local angle = captureAngles[currentAngleIndex]
    
    print(string.format("📷 Capturing view at %d°...", angle))
    
    -- Get RGB image
    local rgbImage = sim.getVisionSensorImage(kinectRGB)
    if rgbImage then
        local rgbFilename = string.format("/tmp/multi_angle_%d_rgb.txt", angle)
        local rgbFile = io.open(rgbFilename, "wb")
        if rgbFile then
            rgbFile:write(rgbImage)
            rgbFile:close()
            print("  ✅ RGB saved:", rgbFilename)
        else
            print("  ❌ Failed to save RGB file:", rgbFilename)
        end
    else
        print("  ❌ Failed to capture RGB image")
    end
    
    -- Get depth image  
    local depthImage = sim.getVisionSensorDepthBuffer(kinectDepth)
    if depthImage then
        local depthFilename = string.format("/tmp/multi_angle_%d_depth.txt", angle)
        local depthFile = io.open(depthFilename, "w")
        if depthFile then
            -- Convert depth buffer to text format
            for i = 1, #depthImage do
                if i > 1 then
                    depthFile:write(",")
                end
                depthFile:write(string.format("%.6f", depthImage[i]))
            end
            depthFile:close()
            print("  ✅ Depth saved:", depthFilename)
        else
            print("  ❌ Failed to save depth file:", depthFilename)
        end
    else
        print("  ❌ Failed to capture depth image")
    end
    
    captureCount = captureCount + 1
    print(string.format("📊 Capture %d/%d completed", captureCount, #captureAngles))
end

function sysCall_actuation()
    local currentTime = sim.getSimulationTime()
    
    if state == "MOVING" and (currentTime - moveStartTime) > waitTime then
        -- Finished moving and waiting, now capture
        captureCurrentView()
        
        -- Move to next angle
        currentAngleIndex = currentAngleIndex + 1
        state = "CAPTURING"
        
        -- Wait a bit before moving to next position
        sim.wait(1.0)
        moveToNextAngle()
    end
end

function sysCall_cleanup()
    -- Ensure camera returns to original position
    if originalPosition and originalOrientation then
        sim.setObjectPosition(kinectRef, -1, originalPosition)
        sim.setObjectOrientation(kinectRef, -1, originalOrientation)
        print("🔄 Camera returned to original position")
    end
end




