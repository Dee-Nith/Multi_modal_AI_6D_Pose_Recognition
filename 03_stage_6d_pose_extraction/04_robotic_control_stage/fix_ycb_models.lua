-- Fix YCB Models - Make Them Static and Non-Respondable
-- This prevents them from disappearing during simulation

function fixYCBModels()
    print("🔧 Fixing YCB models to prevent disappearance...")
    
    local sceneHandle = sim.getObject('.')
    local childCount = sim.getObjectInt32Parameter(sceneHandle, sim.objintparam_childcount)
    local fixedCount = 0
    
    for i = 0, childCount - 1 do
        local child = sim.getObjectChild(sceneHandle, i)
        local childName = sim.getObjectName(child)
        
        -- Check if it's a textured object (YCB model)
        if string.find(childName, "textured") then
            print("Processing: " .. childName)
            
            -- Make it NON-respondable (this is key!)
            sim.setObjectInt32Parameter(child, sim.objintparam_respondable, 0)
            
            -- Make it static (won't move)
            sim.setObjectInt32Parameter(child, sim.objintparam_static, 1)
            
            -- Make it NON-dynamic (won't be affected by physics)
            sim.setObjectInt32Parameter(child, sim.objintparam_dynamic, 0)
            
            -- Make it a model (persists during simulation)
            sim.setObjectInt32Parameter(child, sim.objintparam_model, 1)
            
            -- Set it to not be deleted during simulation
            sim.setObjectInt32Parameter(child, sim.objintparam_cannotbedel, 1)
            
            fixedCount = fixedCount + 1
            print("✅ Fixed: " .. childName)
        end
    end
    
    print("🎉 Fixed " .. fixedCount .. " YCB models!")
    print("📝 Models are now:")
    print("   - Static (won't move)")
    print("   - Non-respondable (no physics collision)")
    print("   - Non-dynamic (not affected by gravity)")
    print("   - Protected from deletion during simulation")
    print("🚀 Ready to test simulation!")
end

-- Run the fix
fixYCBModels()







