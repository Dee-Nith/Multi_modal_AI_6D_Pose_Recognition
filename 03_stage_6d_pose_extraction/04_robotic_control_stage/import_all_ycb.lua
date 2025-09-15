-- Import All YCB Models Script
-- Copy this entire script into CoppeliaSim

function importAllYCBModels()
    print("🤖 Starting import of all 21 YCB models...")
    
    -- Get current scene
    local sceneHandle = sim.getObject('.')
    
    -- Create main YCB models container
    local ycbContainer = sim.createPureShape(0, 0, {0.1, 0.1, 0.1}, 0, nil)
    sim.setObjectName(ycbContainer, 'YCB_All_Models')
    
    -- Use absolute path
    local basePath = "/Users/nith/Desktop/AI_6D_Pose_recognition/Data_sets/YCB-Video-Base/models/"
    
    -- All YCB model IDs and their display names
    local models = {
        {id='002_master_chef_can', name='MasterChefCan', pos={0.5, 0.2, 0.05}},
        {id='003_cracker_box', name='CrackerBox', pos={0.3, 0.4, 0.05}},
        {id='004_sugar_box', name='SugarBox', pos={0.7, 0.3, 0.05}},
        {id='005_tomato_soup_can', name='TomatoSoupCan', pos={0.2, 0.6, 0.05}},
        {id='006_mustard_bottle', name='MustardBottle', pos={0.8, 0.5, 0.05}},
        {id='007_tuna_fish_can', name='TunaFishCan', pos={0.4, 0.8, 0.05}},
        {id='008_pudding_box', name='PuddingBox', pos={0.6, 0.7, 0.05}},
        {id='009_gelatin_box', name='GelatinBox', pos={0.1, 0.3, 0.05}},
        {id='010_potted_meat_can', name='PottedMeatCan', pos={0.9, 0.4, 0.05}},
        {id='011_banana', name='Banana', pos={0.3, 0.1, 0.05}},
        {id='019_pitcher_base', name='PitcherBase', pos={0.7, 0.8, 0.05}},
        {id='021_bleach_cleanser', name='BleachCleanser', pos={0.2, 0.5, 0.05}},
        {id='024_bowl', name='Bowl', pos={0.8, 0.2, 0.05}},
        {id='025_mug', name='Mug', pos={0.5, 0.6, 0.05}},
        {id='035_power_drill', name='PowerDrill', pos={0.1, 0.7, 0.05}},
        {id='036_wood_block', name='WoodBlock', pos={0.9, 0.6, 0.05}},
        {id='037_scissors', name='Scissors', pos={0.4, 0.2, 0.05}},
        {id='040_large_marker', name='LargeMarker', pos={0.6, 0.1, 0.05}},
        {id='051_large_clamp', name='LargeClamp', pos={0.2, 0.8, 0.05}},
        {id='052_extra_large_clamp', name='ExtraLargeClamp', pos={0.8, 0.7, 0.05}},
        {id='061_foam_brick', name='FoamBrick', pos={0.5, 0.5, 0.05}}
    }
    
    local successCount = 0
    local totalCount = #models
    
    -- Import each model
    for i, model in ipairs(models) do
        local objPath = basePath .. model.id .. "/textured.obj"
        print("Importing " .. i .. "/" .. totalCount .. ": " .. model.name)
        
        local modelHandle = sim.importShape(0, objPath, 0, 0, 0)
        if modelHandle ~= -1 then
            sim.setObjectName(modelHandle, model.name)
            sim.setObjectParent(modelHandle, ycbContainer, true)
            
            -- Position the model
            sim.setObjectPosition(modelHandle, -1, model.pos)
            
            successCount = successCount + 1
            print("✅ Successfully imported: " .. model.name)
        else
            print("❌ Failed to import: " .. model.name)
        end
    end
    
    print("🎉 Import completed!")
    print("📊 Results: " .. successCount .. "/" .. totalCount .. " models imported successfully")
    print("📁 All models are organized under 'YCB_All_Models' container")
    
    if successCount > 0 then
        print("🎯 Ready for robotic grasping experiments!")
    end
end

-- Run import
importAllYCBModels()







