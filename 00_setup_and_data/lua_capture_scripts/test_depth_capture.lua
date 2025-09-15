-- Simple Depth Capture Test
-- Test if we can capture depth data from Kinect

print("Simple Depth Capture Test")
print("=========================")

-- Get Kinect camera handles
local kinectRGB = sim.getObject("./kinect/rgb")
local kinectDepth = sim.getObject("./kinect/depth")

print("Kinect RGB handle: " .. tostring(kinectRGB))
print("Kinect Depth handle: " .. tostring(kinectDepth))

if kinectRGB ~= -1 and kinectDepth ~= -1 then
    print("✅ Found Kinect cameras!")
    
    -- Test RGB capture
    print("\n--- Testing RGB Capture ---")
    local rgbImage = sim.getVisionSensorImg(kinectRGB)
    if rgbImage then
        print("✅ RGB capture successful: " .. #rgbImage .. " pixels")
    else
        print("❌ RGB capture failed")
    end
    
    -- Test depth capture with old API
    print("\n--- Testing Depth Capture (Old API) ---")
    local depthImageOld = sim.getVisionSensorDepthBuffer(kinectDepth)
    if depthImageOld then
        print("✅ Depth capture (old API) successful: " .. #depthImageOld .. " values")
        print("First 5 depth values: " .. depthImageOld[1] .. ", " .. depthImageOld[2] .. ", " .. depthImageOld[3] .. ", " .. depthImageOld[4] .. ", " .. depthImageOld[5])
    else
        print("❌ Depth capture (old API) failed")
    end
    
    -- Test depth capture with new API
    print("\n--- Testing Depth Capture (New API) ---")
    local depthImageNew = sim.getVisionSensorDepth(kinectDepth)
    if depthImageNew then
        print("✅ Depth capture (new API) successful: " .. #depthImageNew .. " values")
        print("First 5 depth values: " .. depthImageNew[1] .. ", " .. depthImageNew[2] .. ", " .. depthImageNew[3] .. ", " .. depthImageNew[4] .. ", " .. depthImageNew[5])
    else
        print("❌ Depth capture (new API) failed")
    end
    
    -- Test saving depth data
    print("\n--- Testing Depth Data Saving ---")
    local testDepthData = depthImageNew or depthImageOld
    if testDepthData then
        -- Convert to string
        local depthString = ""
        for i = 1, math.min(10, #testDepthData) do  -- Just first 10 values for test
            if i > 1 then
                depthString = depthString .. ","
            end
            depthString = depthString .. tostring(testDepthData[i])
        end
        
        print("Depth string (first 10 values): " .. depthString)
        
        -- Try to save
        local testFile = io.open("/tmp/test_depth.txt", "w")
        if testFile then
            testFile:write(depthString)
            testFile:close()
            print("✅ Test depth file saved to /tmp/test_depth.txt")
        else
            print("❌ Failed to save test depth file")
        end
    else
        print("❌ No depth data to save")
    end
    
else
    print("❌ Kinect cameras not found!")
    print("Make sure the Kinect object is in your scene")
    print("Expected path: ./kinect/rgb and ./kinect/depth")
end

print("\nTest completed!")




