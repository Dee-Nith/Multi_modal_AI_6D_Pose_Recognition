-- Grasping Scene Setup Script for CoppeliaSim
-- Imports YCB models and sets up UR5 robot for robotic grasping

function setupGraspingScene()
    print("🤖 Setting up Robotic Grasping Scene...")
    
    -- Create scene container
    local sceneContainer = sim.createPureShape(0, 0, {0.1, 0.1, 0.1}, 0, nil)
    sim.setObjectName(sceneContainer, 'Grasping_Scene')
    
    -- Import YCB models (test with 3 models first)
    local ycbModels = {
        {id='002_master_chef_can', name='Master Chef Can', pos={0.5, 0.2, 0.05}},
        {id='003_cracker_box', name='Cracker Box', pos={0.3, 0.4, 0.05}},
        {id='004_sugar_box', name='Sugar Box', pos={0.7, 0.3, 0.05}}
    }
    
    for i, model in ipairs(ycbModels) do
        local objPath = '../../Data_sets/YCB-Video-Base/models/' .. model.id .. '/textured.obj'
        local modelHandle = sim.importShape(0, objPath, 0, 0, 0)
        
        if modelHandle ~= -1 then
            sim.setObjectName(modelHandle, model.name)
            sim.setObjectPosition(modelHandle, -1, model.pos)
            sim.setObjectParent(modelHandle, sceneContainer, true)
            print('✅ Imported: ' .. model.name)
        else
            print('❌ Failed to import: ' .. model.name)
        end
    end
    
    -- Add UR5 robot from model browser
    print("🤖 Adding UR5 robot...")
    -- Note: This would need to be done manually or via API
    
    print("🎉 Scene setup completed!")
    print("📝 Next steps:")
    print("  1. Add UR5 robot from Model Browser → robots → non-mobile")
    print("  2. Setup RGB-D camera")
    print("  3. Configure physics properties")
    print("  4. Start your grasping experiments!")
end

-- Run setup
setupGraspingScene()







