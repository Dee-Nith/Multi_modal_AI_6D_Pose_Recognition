-- Auto Kinect Capture
-- Automatically captures images without waiting

print("Auto Kinect Capture")
print("===================")

-- Get Kinect camera handles using relative paths
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
            local filename = "auto_kinect_" .. imageNumber .. "_rgb.txt"
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
    print("Starting auto kinect capture...")
    print("Objects to capture: Master Chef can, Cracker box, Mug, Banana, Mustard bottle")
    
    print("Ready to capture images!")
    print("Instructions:")
    print("  1. Adjust camera position/angle")
    print("  2. Move objects if needed")
    print("  3. Run this script to capture one image")
    print("  4. Check the captured image")
    print("  5. Re-run script for next image")
    print("  6. Repeat until you have enough images")
    
    -- Capture one image automatically
    local success = captureImage(1)
    if success then
        print("Capture 1 completed successfully!")
        print("Check the image: auto_kinect_1_rgb.txt")
        print("Re-run this script to capture the next image")
    else
        print("Capture failed")
    end
    
    print("Auto kinect capture completed!")
    
else
    print("Kinect cameras not found!")
    print("Make sure the Kinect object is in your scene")
end




