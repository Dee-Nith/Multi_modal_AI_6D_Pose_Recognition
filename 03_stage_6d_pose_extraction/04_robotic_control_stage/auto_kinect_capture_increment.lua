-- Auto Kinect Capture with Smart Increment
-- Automatically finds the next available number

print("Auto Kinect Capture with Smart Increment")
print("=========================================")

-- Get Kinect camera handles using relative paths
local kinectRGB = sim.getObject("./kinect/rgb")
local kinectDepth = sim.getObject("./kinect/depth")

if kinectRGB ~= -1 and kinectDepth ~= -1 then
    print("Found Kinect cameras!")
    
    -- Function to find next available number
    function findNextNumber()
        local nextNum = 1
        
        -- Check for existing files in /tmp (most reliable)
        for i = 1, 100 do
            local filename = "/tmp/auto_kinect_" .. i .. "_rgb.txt"
            local file = io.open(filename, "r")
            if file then
                file:close()
                nextNum = i + 1
            else
                break
            end
        end
        
        -- Also check current directory
        for i = 1, 100 do
            local filename = "auto_kinect_" .. i .. "_rgb.txt"
            local file = io.open(filename, "r")
            if file then
                file:close()
                if i >= nextNum then
                    nextNum = i + 1
                end
            end
        end
        
        return nextNum
    end
    
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
            
            -- Try multiple save locations with incrementing number
            local saveLocations = {
                "auto_kinect_" .. imageNumber .. "_rgb.txt",
                "/tmp/auto_kinect_" .. imageNumber .. "_rgb.txt",
                "kinect_auto_" .. imageNumber .. "_rgb.txt"
            }
            
            for i, filename in ipairs(saveLocations) do
                print("Trying to save to: " .. filename)
                local file = io.open(filename, "wb")
                if file then
                    file:write(rgbData)
                    file:close()
                    print("Kinect RGB data saved to " .. filename)
                    return true
                else
                    print("Failed to save to: " .. filename)
                end
            end
            
            print("Failed to save to any location")
            return false
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
    
    -- Find next available number automatically
    local nextNumber = findNextNumber()
    print("Next image number: " .. nextNumber)
    
    -- Capture one image automatically
    local success = captureImage(nextNumber)
    if success then
        print("Capture " .. nextNumber .. " completed successfully!")
        print("Check the image file that was created")
        print("Re-run this script to capture the next image")
    else
        print("Capture failed")
    end
    
    print("Auto kinect capture completed!")
    
else
    print("Kinect cameras not found!")
    print("Make sure the Kinect object is in your scene")
end
