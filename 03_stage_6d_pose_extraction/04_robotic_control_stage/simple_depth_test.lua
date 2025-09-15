-- Simple Depth Test - Verify data format
print("Simple Depth Test - Data Format")
print("================================")

local kinectDepth = sim.getObject("./kinect/depth")
print("Kinect Depth handle: " .. tostring(kinectDepth))

if kinectDepth ~= -1 then
    -- Capture depth
    local depthImage = sim.getVisionSensorDepthBuffer(kinectDepth)
    if depthImage then
        print("✅ Depth capture successful: " .. #depthImage .. " values")
        
        -- Show first 10 values with proper formatting
        print("First 10 depth values:")
        for i = 1, 10 do
            print("  " .. i .. ": " .. string.format("%.6f", depthImage[i]))
        end
        
        -- Create a small test file with first 100 values
        local testString = ""
        for i = 1, 100 do
            if i > 1 then
                testString = testString .. ","
            end
            testString = testString .. string.format("%.6f", depthImage[i])
        end
        
        -- Save test file
        local file = io.open("/tmp/simple_depth_test.txt", "w")
        if file then
            file:write(testString)
            file:close()
            print("✅ Test file saved: /tmp/simple_depth_test.txt")
            print("First 5 values in file: " .. string.format("%.6f", depthImage[1]) .. ", " .. string.format("%.6f", depthImage[2]) .. ", " .. string.format("%.6f", depthImage[3]) .. ", " .. string.format("%.6f", depthImage[4]) .. ", " .. string.format("%.6f", depthImage[5]))
        else
            print("❌ Failed to save test file")
        end
    else
        print("❌ Depth capture failed")
    end
else
    print("❌ Kinect depth camera not found")
end

print("Test completed!")




