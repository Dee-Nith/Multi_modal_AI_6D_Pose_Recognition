-- Fixed YCB Model Import with Absolute Path
-- Copy this entire script into CoppeliaSim

function importModels()
    print("🤖 Starting YCB model import...")
    
    -- Get the current directory
    local currentDir = sim.getStringParameter(sim.stringparam_scene_path)
    print("Current scene path: " .. currentDir)
    
    -- Create container
    local container = sim.createPureShape(0, 0, {0.1, 0.1, 0.1}, 0, nil)
    sim.setObjectName(container, 'YCB_Models')
    
    -- Try different path approaches
    local paths = {
        "Data_sets/YCB-Video-Base/models/002_master_chef_can/textured.obj",
        "../../Data_sets/YCB-Video-Base/models/002_master_chef_can/textured.obj",
        "/Users/nith/Desktop/AI_6D_Pose_recognition/Data_sets/YCB-Video-Base/models/002_master_chef_can/textured.obj"
    }
    
    for i, path in ipairs(paths) do
        print("Trying path " .. i .. ": " .. path)
        local can = sim.importShape(0, path, 0, 0, 0)
        if can ~= -1 then
            sim.setObjectName(can, 'Master Chef Can')
            sim.setObjectParent(can, container, true)
            print('✅ Successfully imported: Master Chef Can using path ' .. i)
            break
        else
            print('❌ Failed with path ' .. i)
        end
    end
    
    print('🎉 Import attempt completed!')
end

importModels()







