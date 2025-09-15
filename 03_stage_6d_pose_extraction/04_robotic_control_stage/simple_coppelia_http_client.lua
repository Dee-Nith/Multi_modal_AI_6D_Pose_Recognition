-- Simple CoppeliaSim HTTP Client Script
-- Copy and paste this into CoppeliaSim console

print("🌐 Starting HTTP client for camera data...")

-- Get camera handles
local rgbSensor = sim.getObject("./sensorRGB")
local depthSensor = sim.getObject("./sensorDepth")

if rgbSensor ~= -1 and depthSensor ~= -1 then
    print("✅ Found cameras!")
    
    -- Capture RGB image
    local rgbImage = sim.getVisionSensorImg(rgbSensor)
    if rgbImage then
        print("✅ RGB captured: " .. #rgbImage .. " pixels")
        
        -- Save to file first (as backup)
        local file = io.open("current_rgb.txt", "wb")
        if file then
            file:write(rgbImage)
            file:close()
            print("💾 RGB data saved to current_rgb.txt")
        end
        
        -- Try to send via HTTP (if socket.http is available)
        local success = false
        local http_available = pcall(function()
            local http = require("socket.http")
            local ltn12 = require("ltn12")
            
            local response_body = {}
            local res, code, response_headers = http.request{
                url = "http://localhost:8080/camera",
                method = "POST",
                headers = {
                    ["Content-Type"] = "application/octet-stream",
                    ["Content-Length"] = #rgbImage
                },
                source = ltn12.source.string(rgbImage),
                sink = ltn12.sink.table(response_body)
            }
            
            if res then
                print("✅ Camera data sent to HTTP server!")
                success = true
            else
                print("❌ HTTP request failed")
            end
        end)
        
        if not http_available then
            print("⚠️ HTTP library not available, using file-based method")
            print("💡 The camera data is saved to current_rgb.txt")
            print("💡 The Python server will read this file automatically")
        end
        
    else
        print("❌ Failed to capture RGB")
    end
    
    -- Capture depth image
    local depthImage = sim.getVisionSensorDepth(depthSensor)
    if depthImage then
        print("✅ Depth captured: " .. #depthImage .. " pixels")
        
        -- Save to file
        local file = io.open("current_depth.txt", "wb")
        if file then
            file:write(depthImage)
            file:close()
            print("💾 Depth data saved to current_depth.txt")
        end
    else
        print("❌ Failed to capture depth")
    end
    
    print("🎉 Camera capture completed!")
    print("🚀 The Python server should now detect the new camera data!")
    
else
    print("❌ Cameras not found!")
end




