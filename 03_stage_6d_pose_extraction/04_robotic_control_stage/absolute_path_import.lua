-- YCB Model Import with Absolute Path
-- Copy this entire script into CoppeliaSim

function importYCBModels()
    print("🤖 Starting YCB model import...")
    
    -- Get current scene
    local sceneHandle = sim.getObject('.')
    
    -- Create YCB models container
    local ycbContainer = sim.createPureShape(0, 0, {0.1, 0.1, 0.1}, 0, nil)
    sim.setObjectName(ycbContainer, 'YCB_Models')
    
    -- Use absolute path that we know exists
    local basePath = "/Users/nith/Desktop/AI_6D_Pose_recognition/Data_sets/YCB-Video-Base/models/"
    
    -- Import Master Chef Can
    local masterChefCan = sim.importShape(0, basePath .. "002_master_chef_can/textured.obj", 0, 0, 0)
    if masterChefCan ~= -1 then
        sim.setObjectName(masterChefCan, 'Master Chef Can')
        sim.setObjectParent(masterChefCan, ycbContainer, true)
        print('✅ Imported: Master Chef Can')
    else
        print('❌ Failed to import: Master Chef Can')
    end
    
    -- Import Cracker Box
    local crackerBox = sim.importShape(0, basePath .. "003_cracker_box/textured.obj", 0, 0, 0)
    if crackerBox ~= -1 then
        sim.setObjectName(crackerBox, 'Cracker Box')
        sim.setObjectParent(crackerBox, ycbContainer, true)
        print('✅ Imported: Cracker Box')
    else
        print('❌ Failed to import: Cracker Box')
    end
    
    print('🎉 YCB model import completed!')
end

-- Run import
importYCBModels()







