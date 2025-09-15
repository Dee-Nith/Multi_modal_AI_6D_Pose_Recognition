-- Simple Kinect Capture
-- Fresh start based on working approach

print("Simple Kinect Capture")
print("====================")

-- Get Kinect camera handles using relative paths (this worked before)
local kinectRGB = sim.getObject("./kinect/rgb")
local kinectDepth = sim.getObject("./kinect/depth")

if kinectRGB ~= -1 and kinectDepth ~= -1 then
    print("Found Kinect cameras!")
    
    -- Function to capture and save image
    function captureImage(imageNumber)
        print("Capturing image " .. imageNumber .. "...")
        
        -- Capture RGB image
        local rgbImage = sim.getVisionSensorImg(kinectRGB)
        if rgbImage then
            print("Kinect RGB captured: " .. #rgbImage .. " pixels")
            
            -- Convert table to string if needed
            local rgbData = rgbImage
            if type(rgbImage) == "table" then
                rgbData = table.concat(rgbImage)
                print("Converted table to string for RGB data")
            end
            
            -- Save to file
            local filename = "kinect_capture_" .. imageNumber .. "_rgb.txt"
            local file = io.open(filename, "wb")
            if file then
                file:write(rgbData)
                file:close()
                print("Kinect RGB data saved to " .. filename)
                return true
            else
                print("Failed to save Kinect RGB data")
                return false
            end
        else
            print("Failed to capture Kinect RGB")
            return false
        end
    end
    
    -- Main execution
    print("Starting simple kinect capture...")
    print("Objects to capture: Master Chef can, Cracker box, Mug, Banana, Mustard bottle")
    
    print("Ready to capture images!")
    print("Instructions:")
    print("  1. Adjust camera position/angle")
    print("  2. Move objects if needed")
    print("  3. Press Enter to capture one image")
    print("  4. Repeat for 20-30 images")
    print("  5. Type 'done' to finish")
    
    local captureCount = 0
    local maxCaptures = 30
    
    for i = 1, maxCaptures do
        print("Ready for capture " .. (captureCount + 1) .. " (max " .. maxCaptures .. ")")
        print("Adjust camera position and press Enter to capture...")
        print("Type 'done' to finish capturing")
        
        local input = io.read()
        
        if input and input:lower() == "done" then
            print("Capture stopped by user")
            break
        elseif input then
            local success = captureImage(captureCount + 1)
            if success then
                captureCount = captureCount + 1
                print("Capture " .. captureCount .. " completed!")
            else
                print("Capture failed")
            end
        else
            print("Capture stopped by user")
            break
        end
    end
    
    print("Simple Kinect Capture Summary:")
    print("  Total captures: " .. captureCount)
    
    if captureCount > 0 then
        print("Ready for processing!")
        print("Run 'python process_simple_kinect.py' to process captured data")
    else
        print("No data captured")
    end
    
    print("Simple kinect capture completed!")
    
else
    print("Kinect cameras not found!")
    print("Make sure the Kinect object is in your scene")
end
