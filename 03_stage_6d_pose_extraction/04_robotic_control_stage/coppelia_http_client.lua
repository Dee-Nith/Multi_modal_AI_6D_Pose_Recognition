
-- CoppeliaSim HTTP Client Script
-- Attach this to the sphericalVisionRGBAndDepth object

local http = require("socket.http")
local ltn12 = require("ltn12")

function sysCall_init()
    print("🌐 HTTP client initialized")
    server_url = "http://localhost:8080"
    rgbSensor = sim.getObject("./sensorRGB")
    depthSensor = sim.getObject("./sensorDepth")
end

function sysCall_sensing()
    -- Capture camera data every 10 simulation steps
    if sim.getSimulationTime() % 0.1 < 0.01 then
        if rgbSensor ~= -1 then
            local rgbImage = sim.getVisionSensorImg(rgbSensor)
            if rgbImage then
                -- Send camera data to HTTP server
                local response_body = {}
                local res, code, response_headers = http.request{
                    url = server_url .. "/camera",
                    method = "POST",
                    headers = {
                        ["Content-Type"] = "application/octet-stream",
                        ["Content-Length"] = #rgbImage
                    },
                    source = ltn12.source.string(rgbImage),
                    sink = ltn12.sink.table(response_body)
                }
                
                if res then
                    print("✅ Camera data sent to HTTP server")
                else
                    print("❌ Failed to send camera data")
                end
            end
        end
    end
end

function sysCall_cleanup()
    print("🌐 HTTP client stopped")
end
