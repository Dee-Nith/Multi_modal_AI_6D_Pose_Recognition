-- Auto Kinect RGB + Depth Capture - TMP Version
-- Captures both RGB and depth images from Kinect

print("Auto Kinect RGB + Depth Capture - TMP Version")
print("==============================================")

-- Get Kinect camera handles using relative paths
local kinectRGB = sim.getObject("./kinect/rgb")
local kinectDepth = sim.getObject("./kinect/depth")

if kinectRGB ~= -1 and kinectDepth ~= -1 then
    print("Found Kinect RGB and Depth cameras!")
    
    -- Function to find next available number
    function findNextNumber()
        local nextNum = 1
        
        -- Check for existing files in /tmp
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
        
        return nextNum
    end
    
    -- Function to capture and save RGB image
    function captureRGBImage(imageNumber)
        print("Capturing RGB image " .. imageNumber .. "...")
        
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
            
            -- Save RGB to /tmp
            local filename = "/tmp/auto_kinect_" .. imageNumber .. "_rgb.txt"
            local file = io.open(filename, "wb")
            if file then
                file:write(rgbData)
                file:close()
                print("Kinect RGB data saved to " .. filename)
                return true
            else
                print("Failed to save RGB to: " .. filename)
                return false
            end
        else
            print("Failed to capture Kinect RGB")
            return false
        end
    end
    
    -- Function to capture and save depth image
    function captureDepthImage(imageNumber)
        print("Capturing depth image " .. imageNumber .. "...")
        
        -- Capture depth image using the working old API
        local depthImage = sim.getVisionSensorDepthBuffer(kinectDepth)
        if depthImage then
            print("Kinect depth captured: " .. #depthImage .. " depth values")
            
            -- Save depth data in chunks to avoid memory issues
            local filename = "/tmp/auto_kinect_" .. imageNumber .. "_depth.txt"
            local file = io.open(filename, "w")
            if file then
                -- Write depth values in chunks
                local chunkSize = 1000
                for i = 1, #depthImage do
                    if i > 1 then
                        file:write(",")
                    end
                    file:write(string.format("%.6f", depthImage[i]))
                    
                    -- Flush every chunk to ensure data is written
                    if i % chunkSize == 0 then
                        file:flush()
                    end
                end
                file:close()
                print("Kinect depth data saved to " .. filename)
                return true
            else
                print("Failed to save depth to: " .. filename)
                return false
            end
        else
            print("Failed to capture Kinect depth")
            return false
        end
    end
    
    -- Function to capture both RGB and depth
    function captureBothImages(imageNumber)
        print("Capturing both RGB and depth for image " .. imageNumber .. "...")
        
        -- Capture RGB first
        local rgbSuccess = captureRGBImage(imageNumber)
        
        -- Capture depth second
        local depthSuccess = captureDepthImage(imageNumber)
        
        if rgbSuccess and depthSuccess then
            print("✅ Both RGB and depth captured successfully!")
            return true
        elseif rgbSuccess then
            print("⚠️ RGB captured but depth failed")
            return false
        elseif depthSuccess then
            print("⚠️ Depth captured but RGB failed")
            return false
        else
            print("❌ Both RGB and depth capture failed")
            return false
        end
    end
    
    -- Main execution
    print("Starting auto kinect RGB + depth capture...")
    print("Objects to capture: Master Chef can, Cracker box, Mug, Banana, Mustard bottle")
    
    print("Ready to capture RGB + depth images!")
    print("Instructions:")
    print("  1. Adjust camera position/angle")
    print("  2. Move objects if needed")
    print("  3. Run this script to capture one RGB + depth pair")
    print("  4. Check the captured images")
    print("  5. Re-run script for next image pair")
    print("  6. Repeat until you have enough images")
    
    -- Find next available number automatically
    local nextNumber = findNextNumber()
    print("Next image number: " .. nextNumber)
    
    -- Capture both RGB and depth images automatically
    local success = captureBothImages(nextNumber)
    if success then
        print("Capture " .. nextNumber .. " completed successfully!")
        print("Files created in /tmp/:")
        print("  - /tmp/auto_kinect_" .. nextNumber .. "_rgb.txt")
        print("  - /tmp/auto_kinect_" .. nextNumber .. "_depth.txt")
        print("Check the image files that were created")
        print("Re-run this script to capture the next image pair")
    else
        print("Capture failed")
    end
    
    print("Auto kinect RGB + depth capture completed!")
    
else
    print("Kinect cameras not found!")
    print("Make sure the Kinect object is in your scene")
    print("Expected path: ./kinect/rgb and ./kinect/depth")
end




