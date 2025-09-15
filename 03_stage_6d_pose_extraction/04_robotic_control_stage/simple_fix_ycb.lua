-- Simple Fix for YCB Models - Safe Version
-- This prevents crashes and fixes model disappearance

function simpleFixYCB()
    print("🔧 Simple fix for YCB models...")
    
    local sceneHandle = sim.getObject('.')
    local childCount = sim.getObjectInt32Parameter(sceneHandle, sim.objintparam_childcount)
    local fixedCount = 0
    
    print("Found " .. childCount .. " total objects")
    
    -- Process only first 5 models to avoid crashes
    for i = 0, math.min(childCount - 1, 4) do
        local child = sim.getObjectChild(sceneHandle, i)
        local childName = sim.getObjectName(child)
        
        print("Checking: " .. childName)
        
        -- Check if it's a textured object (YCB model)
        if string.find(childName, "textured") then
            print("Fixing: " .. childName)
            
            -- Simple fix: make it static and non-respondable
            sim.setObjectInt32Parameter(child, sim.objintparam_respondable, 0)
            sim.setObjectInt32Parameter(child, sim.objintparam_static, 1)
            
            fixedCount = fixedCount + 1
            print("✅ Fixed: " .. childName)
        end
    end
    
    print("🎉 Fixed " .. fixedCount .. " YCB models safely!")
    print("🚀 Ready to test simulation!")
end

-- Run the simple fix
simpleFixYCB()







