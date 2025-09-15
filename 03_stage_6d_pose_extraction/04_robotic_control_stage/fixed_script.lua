-- Fixed YCB Model Import Script
-- Copy this entire script into CoppeliaSim

function importYCBModels()
    print("🤖 Starting YCB model import...")
    
    -- Get current scene
    local sceneHandle = sim.getObject('.')
    
    -- Create YCB models container
    local ycbContainer = sim.createPureShape(0, 0, {0.1, 0.1, 0.1}, 0, nil)
    sim.setObjectName(ycbContainer, 'YCB_Models')
    sim.setObjectInt32Parameter(ycbContainer, sim.objintparam_visibility_layer, 0)
    
    -- Import models with valid variable names
    -- Import Master Chef Can
    local masterChefCan = sim.importShape(0, '../../Data_sets/YCB-Video-Base/models/002_master_chef_can/textured.obj', 0, 0, 0)
    if masterChefCan ~= -1 then
        sim.setObjectName(masterChefCan, 'Master Chef Can')
        sim.setObjectParent(masterChefCan, ycbContainer, true)
        print('✅ Imported: Master Chef Can')
    else
        print('❌ Failed to import: Master Chef Can')
    end
    
    -- Import Cracker Box
    local crackerBox = sim.importShape(0, '../../Data_sets/YCB-Video-Base/models/003_cracker_box/textured.obj', 0, 0, 0)
    if crackerBox ~= -1 then
        sim.setObjectName(crackerBox, 'Cracker Box')
        sim.setObjectParent(crackerBox, ycbContainer, true)
        print('✅ Imported: Cracker Box')
    else
        print('❌ Failed to import: Cracker Box')
    end
    
    -- Import Sugar Box
    local sugarBox = sim.importShape(0, '../../Data_sets/YCB-Video-Base/models/004_sugar_box/textured.obj', 0, 0, 0)
    if sugarBox ~= -1 then
        sim.setObjectName(sugarBox, 'Sugar Box')
        sim.setObjectParent(sugarBox, ycbContainer, true)
        print('✅ Imported: Sugar Box')
    else
        print('❌ Failed to import: Sugar Box')
    end
    
    print('🎉 YCB model import completed!')
    print('📁 Models are organized under YCB_Models container')
end

-- Run import
importYCBModels()







