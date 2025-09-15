-- Preserve YCB Models During Simulation
-- Copy this script into CoppeliaSim

function preserveYCBModels()
    print("🔧 Preserving YCB models during simulation...")
    
    -- Get the YCB container
    local ycbContainer = sim.getObject('./YCB_All_Models')
    if ycbContainer == -1 then
        print("❌ YCB_All_Models container not found")
        return
    end
    
    -- Get all child objects (the YCB models)
    local childCount = sim.getObjectInt32Parameter(ycbContainer, sim.objintparam_childcount)
    print("Found " .. childCount .. " YCB models")
    
    for i = 0, childCount - 1 do
        local child = sim.getObjectChild(ycbContainer, i)
        local childName = sim.getObjectName(child)
        
        -- Set object properties to preserve during simulation
        sim.setObjectInt32Parameter(child, sim.objintparam_respondable, 0)  -- Not respondable
        sim.setObjectInt32Parameter(child, sim.objintparam_static, 1)       -- Static
        sim.setObjectInt32Parameter(child, sim.objintparam_dynamic, 0)      -- Not dynamic
        sim.setObjectInt32Parameter(child, sim.objintparam_visibility_layer, 0)  -- Visible layer
        
        print("✅ Preserved: " .. childName)
    end
    
    print("🎉 All YCB models preserved for simulation!")
end

-- Run preservation
preserveYCBModels()







